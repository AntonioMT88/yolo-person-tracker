import questionary
import cv_player


config = {
        "yolo_model": "yolov8n",
        "video_source": "",
        "run_on_gpu": True
    }

yolos = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

def main():
    show_main_menu()

    print(config)

    if config["video_source"] != "":
        cv_player.start_cv_player(config)


def show_main_menu():
    while True:
        choice = questionary.select(
            "CONFIGURAZIONE APP:",
            choices=[
                "1. Select YOLOv8 model",
                "2. Select video source",
                "3. Enable GPU acceleration?"
            ]
        ).ask()

        if choice.startswith("1"):
            show_yolo_menu()
        elif choice.startswith("2"):
            show_camera_menu()
        elif choice.startswith("3"):
            selected_response = questionary.select("Would you like to use the GPU for inference?", choices=["Yes (recommanded)", "No"]).ask()
            config["run_on_gpu"] = selected_response.startswith("Yes")
            break

def show_camera_menu():
    source_type = questionary.select(
        "Choose video source type:",
        choices=[
            "Webcam (local camera)",
            "IP Camera (RTSP/HTTP)",
            "Video file (e.g. .mp4)"
        ]
    ).ask()

    if source_type.startswith("Webcam"):
        config["video_source"] = "webcam"
    elif source_type.startswith("IP Camera"):
        config["video_source"] = questionary.text("Enter full IP camera stream URL (e.g. rtsp://...)").ask()
    elif source_type.startswith("Video file"):
        config["video_source"] = questionary.text("Enter path to the video file:").ask()

def show_yolo_menu():
    while True:
        choice = questionary.select(
            "CONFIGURAZIONE APP:",
            choices=[
                "1. yolov8n (Nano): smallest size, lowest accuracy, highest speed",
                "2. yolov8s (Small): small size, moderate accuracy, high speed",
                "3. yolov8m (Medium): balanced size, good accuracy, good speed",
                "4. yolov8l (Large): large size, high accuracy, lower speed",
                "5. yolov8x (Extra Large): largest size, highest accuracy, slowest speed",
            ]
        ).ask()

        if choice.startswith("1"):
            config["yolo_model"] = "yolov8n"
            break
        elif choice.startswith("2"):
            config["yolo_model"] = "yolov8s"
            break
        elif choice.startswith("3"):
            config["yolo_model"] = "yolov8m"
            break
        elif choice.startswith("4"):
            config["yolo_model"] = "yolov8l"
            break
        elif choice.startswith("5"):
            config["yolo_model"] = "yolov8x"
            break


if __name__ == "__main__":
    main()