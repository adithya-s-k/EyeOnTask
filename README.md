# Eye On Task

Eye On Task is a project that focuses on two main components: iris detection/blink detection and expression detection. This README file provides an overview of the project and instructions on how to use the code effectively.

## Features

The Eye On Task project consists of the following features:

1. Iris Detection/Blink Detection: This feature analyzes the distance ratio between the upper eyelid and lower eyelid to determine if a person is blinking. Additionally, it tracks the movement of the iris to determine if the user is looking at the screen. The concentration score is calculated based on the user's eye movement throughout the session.

2. Expression Detection: The GUI (Graphical User Interface) provides an option to add expressions and train a model for expression detection. After training the model, the system can classify expressions in real-time using a live camera feed.

3. Application in Test Centers: Eye On Task can be utilized in test centers to monitor students' concentration and detect any unusual eye movement or expressions that might indicate cheating or distraction.

## Getting Started

To use the Eye On Task project, follow the instructions below:

1. Clone the repository from GitHub: [repository-link](https://github.com/adithya-s-k/EyeOnTask).

2. Install the necessary dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Launch the GUI by executing the `main.py` file:
   ```
   python main.py
   ```

4. Once the GUI is open, you will see options to train the expression detection model and perform inference using the trained model.

## Training Expression Detection Model

To train the expression detection model, follow these steps:

1. Click on the "Train Expression Model" button in the GUI.

2. A new window will open, allowing you to add expressions and corresponding labeled data.

3. Capture images for each expression by following the on-screen instructions.

4. After capturing sufficient data, click on the "Train" button to train the model.

5. Once the training is complete, the model will be saved for future use.

## Performing Inference

To perform expression detection using the trained model, follow these steps:

1. Ensure that the trained model is available in the project directory.

2. Click on the "Perform Inference" button in the GUI.

3. The live camera feed will be displayed, and the system will classify expressions in real-time.

4. The detected expressions will be shown on the screen.

## Important Note

- Please ensure that your camera is properly connected and functional before using the Eye On Task project.

- The accuracy of expression detection and blink detection depends on various factors, including lighting conditions and camera quality. Ensure that the environment is well-lit and the camera provides clear and reliable input.

- For test center applications, make sure to comply with relevant privacy and data protection regulations.

## Contributing

We welcome contributions to the Eye On Task project. If you find any issues or have suggestions for improvement, please open an issue on our GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or inquiries about the Eye On Task project, please contact us at [email address].

---

This README file provides an overview of the Eye On Task project and instructions on how to use the code effectively. We hope you find it useful and enjoy using Eye On Task!