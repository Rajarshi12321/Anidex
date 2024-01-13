
# Anidex

Welcome to the Anidex repository, which is a real life pokedex for knowing about animals on the spot with scanning or clicking the picture of the animal.
Then the app will respond which possible animal is it with few basic details in both text in screen and as voice message like any voice assistant <be>

You need to setup the API keys for ANIMAL_API_TOKEN from the [free live animal api](https://api-ninjas.com/api/animals) you can  use it for free and HUGGINGFACEHUB_APT_TOKEN from [HUGGINGFACEHUB](https://huggingface.co/)

## Table of Contents

- [Anidex](#anidex)
  - [Table of Contents](#table-of-contents)
  - [Installation and Dependencies](#installation-and-dependencies)
  - [Working with the code](#working-with-the-code)
  - [Image](#image)
  - [Contributing](#contributing)
  - [License](#license)


## Installation and Dependencies

These are some required packages for our program which are mentioned in the Requirements.txt file

- flask
- fastapi
- python-multipart
- pillow
- tensorflow-serving-api
- matplotlib
- numpy
- requests
- python-dotenv
- langchain
- speechRecognition
- pyttsx3
- tensorflow
- uvicorn
- typing-inspect
- typing_extensions





## Working with the code


I have commented most of the neccesary information in the respective files.

To run this project locally, please follow these steps:-

1. Clone the repository:

   ```shell
   git clone https://github.com/Rajarshi12321/Anidex.git
   ```


2. Activating the env
  
   ```shell
   conda activate <your-env-name> 
   ```

3. Install the required dependencies by running:
   ```shell
   pip install -r requirements.txt
   ``` 
   Ensure you have Python installed on your system (Python 3.9 or higher is recommended).<br />
   Once the dependencies are installed, you're ready to use the project.



4. Run the Flask app: Execute the following code in your terminal.
   ```shell  
   python app.py 
   ```
   

5. Access the app: Open your web browser and navigate to http://localhost:5000/ to use the Anidex app for uploading image file.


## Image

Home Page:
![image](https://github.com/Rajarshi12321/Anidex/assets/94736350/a011627b-70aa-4daa-a8ab-e8c8e76bb588)

Prediction Result Page:
![image](https://github.com/Rajarshi12321/Anidex/assets/94736350/7d02e7d1-fe51-4a8c-9f33-5d46ad2eb6ff)


[![Watch the video](<img width="529" alt="image" src="https://github.com/Rajarshi12321/Anidex/assets/94736350/d148e2f2-fc00-43f9-a717-671bd7e23b95">)](https://youtu.be/geGlNDZqEJ8)



## Contributing
I welcome contributions to improve the functionality and performance of the app. If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bug fix.

2. Make your changes and ensure that the code is well-documented.

3. Test your changes thoroughly to maintain app reliability.

4. Create a pull request, detailing the purpose and changes made in your contribution.



## License
This project is licensed under the MIT License. Feel free to modify and distribute it as per the terms of the license.

I hope this README provides you with the necessary information to get started with the Anidex project. 

