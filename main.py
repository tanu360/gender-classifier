import os
import sys
import subprocess
import importlib.util
import shutil
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any


def configure_native_runtime():
    """Keep native ML/image libraries from oversubscribing CPU threads."""
    thread_env_defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    for name, value in thread_env_defaults.items():
        os.environ.setdefault(name, value)


configure_native_runtime()


def install(package, import_name=None):
    check_name = import_name if import_name else package
    if importlib.util.find_spec(check_name) is None:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


# Dependencies: (pip_name, import_name)
deps = [
    ("deepface", "deepface"),
    ("colorama", "colorama"),
    ("emoji", "emoji"),
    ("pillow", "PIL"),
    ("tf-keras", "tf_keras"),
    ("opencv-python", "cv2"),
]

for pkg, imp in deps:
    install(pkg, imp)
import emoji  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from deepface import DeepFace  # noqa: E402
from colorama import init, Fore, Style  # noqa: E402
from PIL import Image  # noqa: E402


# Initialize colorama
init(autoreset=True)

# Global configuration
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
DEFAULT_WORKERS = 1
CATEGORY_FILENAME_PREFIXES = {
    "male": "male",
    "female": "female",
    "no_human": "no-human",
    "error": "error",
}


def get_worker_count():
    """Read worker count from env, defaulting to the stable single-worker path."""
    raw_value = os.environ.get("GENDER_CLASSIFIER_WORKERS", str(DEFAULT_WORKERS))
    try:
        return max(1, int(raw_value))
    except ValueError:
        print(
            Fore.YELLOW
            + f"{EMOJI['warning']} Invalid GENDER_CLASSIFIER_WORKERS={raw_value!r}; using {DEFAULT_WORKERS}."
        )
        return DEFAULT_WORKERS


def emj(name):
    return emoji.emojize(f":{name}:", language="alias")


# Console icon set kept in one place so output text stays readable.
EMOJI = {
    "target": emj("dart"),
    "family": f"{emj('man')}\u200d{emj('woman')}",
    "robot": emj("robot_face"),
    "speed": emj("zap"),
    "folder": emj("file_folder"),
    "open_folder": emj("open_file_folder"),
    "point": emj("point_right"),
    "error": emj("x"),
    "success": emj("white_check_mark"),
    "warning": emj("warning"),
    "boom": emj("boom"),
    "rocket": emj("rocket"),
    "chart": emj("bar_chart"),
    "male": emj("man"),
    "female": emj("woman"),
    "no_human": emj("no_entry_sign"),
    "party": emj("tada"),
    "stopwatch": emj("stopwatch"),
    "clock": emj("clock1"),
    "thinking": emj("thinking"),
    "wave": emj("wave"),
}


class GenderClassifier:
    def __init__(self):
        self.male_counter = 0
        self.female_counter = 0
        self.no_human_counter = 0
        self.error_counter = 0
        self.total_processed = 0
        self.counter_lock = Lock()
        self.file_counters = {
            "male": 0,
            "female": 0,
            "no_human": 0,
            "error": 0,
        }
        self.analysis_lock = Lock()

    def print_banner(self):
        print(
            Fore.MAGENTA
            + Style.BRIGHT
            + f"\n{EMOJI['target']} AI Gender Classification Tool {EMOJI['family']}\n"
        )
        print(
            Fore.CYAN
            + f"{EMOJI['robot']} Powered by DeepFace AI Engine {EMOJI['robot']}"
        )
        print(
            Fore.CYAN
            + f"{EMOJI['speed']} Crash-safe AI processing {EMOJI['speed']}"
        )
        print(Fore.CYAN + f"{EMOJI['folder']} Auto-organizes images by gender\n")

    def get_input_folder(self):
        """Get input folder from user with validation"""
        print(Fore.YELLOW + f"{EMOJI['open_folder']} Input Folder Selection")

        while True:
            folder_path = input(
                Fore.CYAN
                + f"{EMOJI['point']} Enter the folder path containing images: "
            ).strip()

            if not folder_path:
                print(Fore.RED + f"{EMOJI['error']} Please enter a valid folder path.")
                continue

            if not os.path.exists(folder_path):
                print(Fore.RED + f"{EMOJI['error']} Folder not found: {folder_path}")
                continue

            if not os.path.isdir(folder_path):
                print(
                    Fore.RED
                    + f"{EMOJI['error']} Path is not a directory: {folder_path}"
                )
                continue

            # Check for images in folder
            image_files = self.get_image_files(folder_path)
            if not image_files:
                print(
                    Fore.RED
                    + f"{EMOJI['error']} No supported image files found in this folder."
                )
                print(
                    Fore.YELLOW
                    + f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                )
                continue

            print(
                Fore.GREEN
                + f"{EMOJI['success']} Found {len(image_files)} images in folder"
            )
            return folder_path, image_files

    def get_image_files(self, folder_path):
        """Get all supported image files from folder"""
        image_files = []
        output_base = os.path.abspath(os.path.join(folder_path, "classified_images"))

        for root, dirs, files in os.walk(folder_path):
            if self.is_inside_path(root, output_base):
                dirs[:] = []
                continue

            dirs[:] = [
                dirname
                for dirname in dirs
                if not self.is_inside_path(os.path.join(root, dirname), output_base)
            ]

            for filename in files:
                if filename.lower().endswith(SUPPORTED_FORMATS):
                    image_files.append(os.path.join(root, filename))

        return sorted(list(set(image_files)))  # Remove duplicates and sort

    @staticmethod
    def is_inside_path(path, parent_path):
        """Return True when path is inside parent_path."""
        try:
            return (
                os.path.commonpath([os.path.abspath(path), os.path.abspath(parent_path)])
                == os.path.abspath(parent_path)
            )
        except ValueError:
            return False

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

        self.initialize_file_counters(folders)

        print(
            Fore.GREEN + f"{EMOJI['folder']} Output folders created at: {output_base}"
        )
        return folders

    def initialize_file_counters(self, output_folders):
        """Start output filenames after any existing category-prefixed files."""
        self.file_counters = {
            "male": self.get_highest_existing_index(
                output_folders["male"], CATEGORY_FILENAME_PREFIXES["male"]
            ),
            "female": self.get_highest_existing_index(
                output_folders["female"], CATEGORY_FILENAME_PREFIXES["female"]
            ),
            "no_human": self.get_highest_existing_index(
                output_folders["no_human"], CATEGORY_FILENAME_PREFIXES["no_human"]
            ),
            "error": self.get_highest_existing_index(
                output_folders["errors"], CATEGORY_FILENAME_PREFIXES["error"]
            ),
        }

    @staticmethod
    def get_highest_existing_index(folder_path, filename_prefix):
        """Find the highest index from files like male-1.jpg."""
        highest_index = 0
        expected_prefix = f"{filename_prefix}-"
        for filename in os.listdir(folder_path):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in SUPPORTED_FORMATS:
                continue
            if not stem.startswith(expected_prefix):
                continue

            index_text = stem.removeprefix(expected_prefix)
            if index_text.isdigit():
                highest_index = max(highest_index, int(index_text))
        return highest_index

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

                with self.analysis_lock:
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
                            print(
                                Fore.YELLOW
                                + f"{EMOJI['warning']} Failed to analyze with {detector}: {e}"
                            )
                            continue  # Try next detector

                # If both detectors failed
                if result is None:
                    return self.move_image(
                        image_path, output_folders["no_human"], "no_human"
                    )

                # Handle both single face and multiple faces.
                analysis: dict[str, Any] | None = None
                if isinstance(result, list):
                    if not result:
                        return self.move_image(
                            image_path, output_folders["no_human"], "no_human"
                        )
                    first_result = result[0]
                    if isinstance(first_result, dict):
                        analysis = first_result
                elif isinstance(result, dict):
                    analysis = result

                if analysis is None:
                    return self.move_image(
                        image_path, output_folders["errors"], "error"
                    )

                gender = str(analysis.get("dominant_gender", "")).lower()

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
        """Copy image to destination folder with thread-safe sequential naming"""
        try:
            file_ext = os.path.splitext(src_path)[1].lower()

            with self.counter_lock:
                filename_prefix = CATEGORY_FILENAME_PREFIXES[category]
                next_index = self.file_counters[category] + 1
                while True:
                    new_filename = f"{filename_prefix}-{next_index}{file_ext}"
                    dest_path = os.path.join(dest_folder, new_filename)
                    if not os.path.exists(dest_path):
                        break
                    next_index += 1

                shutil.copy2(src_path, dest_path)
                self.file_counters[category] = next_index

                if category == "male":
                    self.male_counter += 1
                elif category == "female":
                    self.female_counter += 1
                elif category == "no_human":
                    self.no_human_counter += 1
                else:
                    self.error_counter += 1

                self.total_processed += 1
                processed_count = self.total_processed

            return {
                "success": True,
                "category": category,
                "src": os.path.basename(src_path),
                "dest": new_filename,
                "processed_count": processed_count,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "category": "error",
                "src": os.path.basename(src_path),
                "dest": None,
                "processed_count": self.total_processed,
                "error": str(e),
            }

    def process_images_parallel(self, image_files, output_folders):
        """Process images with a stable default and optional guarded threads."""
        print(Fore.CYAN + f"\n{EMOJI['robot']} Initializing AI models...")

        # Pre-download model by testing on a dummy image to avoid parallel download conflicts
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                test_path = temp_file.name

            try:
                Image.fromarray(test_img).save(test_path)

                # This will download the model if not already present
                with self.analysis_lock:
                    DeepFace.analyze(
                        test_path,
                        actions=["gender"],
                        enforce_detection=False,
                        silent=True,
                        detector_backend="opencv",
                    )
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)

            print(Fore.GREEN + f"{EMOJI['success']} AI models ready!")
        except Exception as e:
            print(Fore.YELLOW + f"{EMOJI['warning']}  Model initialization failed: {e}")
            print(
                Fore.RED
                + f"{EMOJI['boom']} Cannot proceed without AI models. Exiting..."
            )
            return  # Exit the function early

        worker_count = get_worker_count()
        if worker_count == 1:
            print(
                Fore.CYAN
                + f"{EMOJI['rocket']} Starting crash-safe processing with 1 worker..."
            )
        else:
            print(
                Fore.CYAN
                + f"{EMOJI['rocket']} Starting guarded threaded processing with {worker_count} workers..."
            )
        print(
            Fore.YELLOW
            + f"{EMOJI['chart']} Total images to process: {len(image_files)}\n"
        )

        start_time = time.time()

        if worker_count == 1:
            for image_path in image_files:
                result = self.classify_single_image(image_path, output_folders)
                self.print_progress(result, len(image_files))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
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
                            + f"{EMOJI['error']} Error processing {os.path.basename(image_path)}: {e}"
                        )

        end_time = time.time()
        self.print_summary(len(image_files), end_time - start_time)

    def print_progress(self, result, total_images):
        """Print progress for each processed image"""
        processed_count = result.get("processed_count", self.total_processed)

        if result["success"]:
            if result["category"] == "male":
                icon = EMOJI["male"]
                color = Fore.BLUE
            elif result["category"] == "female":
                icon = EMOJI["female"]
                color = Fore.MAGENTA
            elif result["category"] == "no_human":
                icon = EMOJI["no_human"]
                color = Fore.YELLOW
            else:
                icon = EMOJI["error"]
                color = Fore.RED

            print(
                f"{color}{icon} [{processed_count}/{total_images}] {result['src']} → {result['dest']}"
            )
        else:
            print(
                Fore.RED
                + f"{EMOJI['error']} [{processed_count}/{total_images}] Failed: {result['src']} - {result['error']}"
            )

    def print_summary(self, total_images, elapsed_time):
        """Print final processing summary"""
        print(Fore.GREEN + Style.BRIGHT + "\n" + "=" * 60)
        print(
            Fore.GREEN
            + Style.BRIGHT
            + f"{EMOJI['party']} CLASSIFICATION COMPLETE! {EMOJI['party']}"
        )
        print(Fore.GREEN + Style.BRIGHT + "=" * 60)

        print(Fore.CYAN + f"{EMOJI['chart']} Processing Summary:")
        print(Fore.BLUE + f"   {EMOJI['male']} Male images: {self.male_counter}")
        print(
            Fore.MAGENTA + f"   {EMOJI['female']} Female images: {self.female_counter}"
        )
        print(
            Fore.YELLOW
            + f"   {EMOJI['no_human']} No human detected: {self.no_human_counter}"
        )
        print(Fore.RED + f"   {EMOJI['error']} Errors: {self.error_counter}")
        print(
            Fore.GREEN
            + f"   {EMOJI['success']} Total processed: {self.total_processed}"
        )

        print(Fore.CYAN + f"\n{EMOJI['stopwatch']}  Performance Stats:")
        print(Fore.CYAN + f"   {EMOJI['clock']} Total time: {elapsed_time:.2f} seconds")
        print(
            Fore.CYAN
            + f"   {EMOJI['speed']} Average per image: {elapsed_time / total_images:.3f} seconds"
        )
        print(
            Fore.CYAN
            + f"   {EMOJI['rocket']} Images per second: {total_images / elapsed_time:.2f}"
        )

        # Calculate accuracy
        successfully_classified = self.male_counter + self.female_counter
        if total_images > 0:
            success_rate = (successfully_classified / total_images) * 100
            print(
                Fore.GREEN + f"   {EMOJI['target']} Success rate: {success_rate:.1f}%"
            )

    def run(self):
        """Main execution method"""
        try:
            self.print_banner()

            # Get input folder and image files
            input_folder, image_files = self.get_input_folder()

            # Setup output folders
            output_folders = self.setup_output_folders(input_folder)

            # Confirm processing
            print(
                Fore.YELLOW
                + f"\n{EMOJI['thinking']} Ready to process {len(image_files)} images."
            )
            confirm = (
                input(Fore.CYAN + f"{EMOJI['point']} Continue? (y/n): ").strip().lower()
            )

            if confirm not in ["y", "yes"]:
                print(Fore.YELLOW + f"{EMOJI['wave']} Operation cancelled. Goodbye!")
                return

            # Process images
            self.process_images_parallel(image_files, output_folders)

        except KeyboardInterrupt:
            print(
                Fore.YELLOW
                + f"\n{EMOJI['warning']}  Process interrupted by user. Goodbye!"
            )
        except Exception as e:
            print(Fore.RED + f"\n{EMOJI['boom']} Unexpected error: {e}")


def main():
    classifier = GenderClassifier()
    classifier.run()


if __name__ == "__main__":
    main()
