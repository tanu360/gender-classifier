import os
import sys
import subprocess
import importlib.util
import shutil
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def install(package, import_name=None):
    check_name = import_name if import_name else package
    if importlib.util.find_spec(check_name) is None:
        print(f"❓ Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


# Dependencies: (pip_name, import_name)
deps = [
    ("deepface", "deepface"),
    ("colorama", "colorama"),
    ("pillow", "PIL"),
    ("tf-keras", "tf_keras"),
    ("opencv-python", "cv2"),
]

for pkg, imp in deps:
    install(pkg, imp)
import numpy as np  # noqa: E402

from deepface import DeepFace  # noqa: E402
from colorama import init, Fore, Style  # noqa: E402
from PIL import Image  # noqa: E402


# Initialize colorama
init(autoreset=True)

# Global configuration
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
MAX_WORKERS = 4  # Adjust based on your CPU cores


class GenderClassifier:
    def __init__(self):
        self.male_counter = 0
        self.female_counter = 0
        self.no_human_counter = 0
        self.error_counter = 0
        self.total_processed = 0

    def print_banner(self):
        print(Fore.MAGENTA + Style.BRIGHT + "\n🎯 AI Gender Classification Tool 👨‍👩\n")
        print(Fore.CYAN + "🤖 Powered by DeepFace AI Engine 🤖")
        print(Fore.CYAN + "⚡ Multi-threaded processing for speed ⚡")
        print(Fore.CYAN + "📁 Auto-organizes images by gender\n")

    def get_input_folder(self):
        """Get input folder from user with validation"""
        print(Fore.YELLOW + "📂 Input Folder Selection")

        while True:
            folder_path = input(
                Fore.CYAN + "👉 Enter the folder path containing images: "
            ).strip()

            if not folder_path:
                print(Fore.RED + "❌ Please enter a valid folder path.")
                continue

            if not os.path.exists(folder_path):
                print(Fore.RED + f"❌ Folder not found: {folder_path}")
                continue

            if not os.path.isdir(folder_path):
                print(Fore.RED + f"❌ Path is not a directory: {folder_path}")
                continue

            # Check for images in folder
            image_files = self.get_image_files(folder_path)
            if not image_files:
                print(Fore.RED + "❌ No supported image files found in this folder.")
                print(
                    Fore.YELLOW
                    + f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                )
                continue

            print(Fore.GREEN + f"✅ Found {len(image_files)} images in folder")
            return folder_path, image_files

    def get_image_files(self, folder_path):
        """Get all supported image files from folder"""
        image_files = []
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(folder_path, f"**/*{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
            # Also check uppercase extensions
            pattern = os.path.join(folder_path, f"**/*{ext.upper()}")
            image_files.extend(glob.glob(pattern, recursive=True))

        return sorted(list(set(image_files)))  # Remove duplicates and sort

    def setup_output_folders(self, input_folder):
        """Create output folder structure"""
        output_base = os.path.join(input_folder, "classified_images")

        folders = {
            "male": os.path.join(output_base, "male"),
            "female": os.path.join(output_base, "female"),
            "no_human": os.path.join(output_base, "no_human"),
            "errors": os.path.join(output_base, "errors"),
        }

        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        print(Fore.GREEN + f"📁 Output folders created at: {output_base}")
        return folders

    def classify_single_image(self, image_path, output_folders):
        """Classify a single image and move to appropriate folder"""
        try:
            # Validate image file
            try:
                img = Image.open(image_path)
                img.verify()  # Verify it's a valid image
            except Exception:
                return self.move_image(image_path, output_folders["errors"], "error")

            # Analyze with DeepFace
            try:
                # Try opencv first, then retinaface as fallback
                detectors = ["opencv", "retinaface"]
                result = None

                for detector in detectors:
                    try:
                        result = DeepFace.analyze(
                            img_path=image_path,
                            actions=["gender"],
                            enforce_detection=False,
                            silent=True,
                            detector_backend=detector,
                        )
                        break  # Success, exit loop
                    except Exception as e:
                        print(Fore.YELLOW + f"⚠️ Failed to analyze with {detector}: {e}")
                        continue  # Try next detector

                # If both detectors failed
                if result is None:
                    return self.move_image(
                        image_path, output_folders["no_human"], "no_human"
                    )

                # Handle both single face and multiple faces
                if isinstance(result, list):
                    result = result[0]  # Take first detected face

                gender = result["dominant_gender"].lower()

                if gender == "man":
                    return self.move_image(image_path, output_folders["male"], "male")
                elif gender == "woman":
                    return self.move_image(
                        image_path, output_folders["female"], "female"
                    )
                else:
                    return self.move_image(
                        image_path, output_folders["no_human"], "no_human"
                    )

            except Exception as e:
                # No face detected or other DeepFace error
                if (
                    "Face could not be detected" in str(e)
                    or "no face" in str(e).lower()
                ):
                    return self.move_image(
                        image_path, output_folders["no_human"], "no_human"
                    )
                else:
                    return self.move_image(
                        image_path, output_folders["errors"], "error"
                    )

        except Exception:
            return self.move_image(image_path, output_folders["errors"], "error")

    def move_image(self, src_path, dest_folder, category):
        """Move image to destination folder with sequential naming"""
        try:
            # Get file extension
            file_ext = os.path.splitext(src_path)[1].lower()

            # Generate sequential filename
            if category == "male":
                self.male_counter += 1
                new_filename = f"{self.male_counter}{file_ext}"
            elif category == "female":
                self.female_counter += 1
                new_filename = f"{self.female_counter}{file_ext}"
            elif category == "no_human":
                self.no_human_counter += 1
                new_filename = f"{self.no_human_counter}{file_ext}"
            else:  # error
                self.error_counter += 1
                new_filename = f"{self.error_counter}{file_ext}"

            dest_path = os.path.join(dest_folder, new_filename)

            # Copy file to preserve original
            shutil.copy2(src_path, dest_path)

            self.total_processed += 1
            return {
                "success": True,
                "category": category,
                "src": os.path.basename(src_path),
                "dest": new_filename,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "category": "error",
                "src": os.path.basename(src_path),
                "dest": None,
                "error": str(e),
            }

    def process_images_parallel(self, image_files, output_folders):
        """Process images using multiple threads"""
        print(Fore.CYAN + "\n🤖 Initializing AI models...")

        # Pre-download model by testing on a dummy image to avoid parallel download conflicts
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_path = "temp_test.jpg"
            Image.fromarray(test_img).save(test_path)

            # This will download the model if not already present
            DeepFace.analyze(
                test_path, actions=["gender"], enforce_detection=False, silent=True
            )
            os.remove(test_path)  # Clean up
            print(Fore.GREEN + "✅ AI models ready!")
        except Exception as e:
            print(Fore.YELLOW + f"⚠️  Model initialization failed: {e}")
            print(Fore.RED + "💥 Cannot proceed without AI models. Exiting...")
            return  # Exit the function early

        print(
            Fore.CYAN + f"🚀 Starting parallel processing with {MAX_WORKERS} workers..."
        )
        print(Fore.YELLOW + f"📊 Total images to process: {len(image_files)}\n")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(
                    self.classify_single_image, img_path, output_folders
                ): img_path
                for img_path in image_files
            }

            # Process completed tasks
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    self.print_progress(result, len(image_files))
                except Exception as e:
                    print(
                        Fore.RED
                        + f"❌ Error processing {os.path.basename(image_path)}: {e}"
                    )

        end_time = time.time()
        self.print_summary(len(image_files), end_time - start_time)

    def print_progress(self, result, total_images):
        """Print progress for each processed image"""
        if result["success"]:
            if result["category"] == "male":
                icon = "👨"
                color = Fore.BLUE
            elif result["category"] == "female":
                icon = "👩"
                color = Fore.MAGENTA
            elif result["category"] == "no_human":
                icon = "🚫"
                color = Fore.YELLOW
            else:
                icon = "❌"
                color = Fore.RED

            print(
                f"{color}{icon} [{self.total_processed}/{total_images}] {result['src']} → {result['dest']}"
            )
        else:
            print(
                Fore.RED
                + f"❌ [{self.total_processed}/{total_images}] Failed: {result['src']} - {result['error']}"
            )

    def print_summary(self, total_images, elapsed_time):
        """Print final processing summary"""
        print(Fore.GREEN + Style.BRIGHT + "\n" + "=" * 60)
        print(Fore.GREEN + Style.BRIGHT + "🎉 CLASSIFICATION COMPLETE! 🎉")
        print(Fore.GREEN + Style.BRIGHT + "=" * 60)

        print(Fore.CYAN + "📊 Processing Summary:")
        print(Fore.BLUE + f"   👨 Male images: {self.male_counter}")
        print(Fore.MAGENTA + f"   👩 Female images: {self.female_counter}")
        print(Fore.YELLOW + f"   🚫 No human detected: {self.no_human_counter}")
        print(Fore.RED + f"   ❌ Errors: {self.error_counter}")
        print(Fore.GREEN + f"   ✅ Total processed: {self.total_processed}")

        print(Fore.CYAN + "\n⏱️  Performance Stats:")
        print(Fore.CYAN + f"   🕐 Total time: {elapsed_time:.2f} seconds")
        print(
            Fore.CYAN
            + f"   ⚡ Average per image: {elapsed_time / total_images:.3f} seconds"
        )
        print(Fore.CYAN + f"   🚀 Images per second: {total_images / elapsed_time:.2f}")

        # Calculate accuracy
        successfully_classified = self.male_counter + self.female_counter
        if total_images > 0:
            success_rate = (successfully_classified / total_images) * 100
            print(Fore.GREEN + f"   🎯 Success rate: {success_rate:.1f}%")

    def run(self):
        """Main execution method"""
        try:
            self.print_banner()

            # Get input folder and image files
            input_folder, image_files = self.get_input_folder()

            # Setup output folders
            output_folders = self.setup_output_folders(input_folder)

            # Confirm processing
            print(Fore.YELLOW + f"\n🤔 Ready to process {len(image_files)} images.")
            confirm = input(Fore.CYAN + "👉 Continue? (y/n): ").strip().lower()

            if confirm not in ["y", "yes"]:
                print(Fore.YELLOW + "👋 Operation cancelled. Goodbye!")
                return

            # Process images
            self.process_images_parallel(image_files, output_folders)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n⚠️  Process interrupted by user. Goodbye!")
        except Exception as e:
            print(Fore.RED + f"\n💥 Unexpected error: {e}")


def main():
    classifier = GenderClassifier()
    classifier.run()


if __name__ == "__main__":
    main()
