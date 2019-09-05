# Security App with Object Detection

This app will use your [Google Coral ML accelerator](https://coral.withgoogle.com) to check for objects (you can choose between people, cats, and dogs). Here is what happens when you run the app:

1. You will see a frame from your camera, you can use your mouse to select a region you would like to keep an eye on

2. Based on your provided input, the app will keep an eye out for people, cats, or dogs.
3. If the app is alerted to an object, it will keep track of the time and alert in a pandas dataframe and also grab a snapshot so you have evidence!
4. When you get back home, press the 'q' key on your keyboard to exit the app and save the dataframe as a CSV



## Requirements:

You will need:

- Python 3 

- OpenCV ([installation guide](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/))

- EdgeTPU detection engine ([installation guide](https://coral.withgoogle.com/docs/accelerator/get-started/))

- An actual Google Coral, preferably plugged in to a USB 3 port but USB 2 should perform fine

- A couple of python packages:

  - ```bash
    pip3 install --user numpy, pandas
    ```



## Running:

Run the app with the simple command:

```bash
python3 security.py "dog"
```

You can also choose "cat" as the detection option. If you don't choose anything the app will by default look for people.

Remember to exit the app by pressing 'q'. 