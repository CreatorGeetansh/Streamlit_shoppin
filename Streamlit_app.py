import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from io import BytesIO
import torch

indian_actors = [
    'Aamir Khan', 'Abhay Deol', 'Abhishek Bachchan', 'Adah Sharma', 'Aditi Rao Hydari',
    'Aditya Roy Kapoor', 'Aftab Shivdasani', 'Aishwarya Rai', 'Ajay Devgn', 'Akshay Kumar',
    'Akshaye Khanna', 'Alia Bhatt', 'Allu Arjun', 'Ameesha Patel', 'Amitabh Bachchan',
    'Amrish Puri', 'Amrita Rao', 'Amy Jackson', 'Ananya Panday', 'Anil Kapoor',
    'Anupam Kher', 'Anushka Sharma', 'Anushka Shetty', 'Arjun Kapoor', 'Arjun Rampal',
    'Arshad Warsi', 'Asin', 'Ayesha Takia', 'Ayushmann Khurrana', 'Bhumi Pednekar',
    'Bipasha Basu', 'Bobby Deol', 'Boman Irani', 'Chiranjeevi', 'Chitrangda Singh',
    'Chunky Pandey', 'Deepika Padukone', 'Dhanush', 'Dia Mirza', 'Disha Patani',
    'Dulquer Salmaan', 'Emraan Hashmi', 'Esha Gupta', 'Farhan Akhtar', 'Fatima Sana Shaikh',
    'Govinda', 'Gul Panag', 'Hrithik Roshan', 'Huma Qureshi', 'Ileana DCruz',
    'Irrfan Khan', 'Jacqueline Fernandez', 'Janvi Kapoor', 'Jimmy Shergill', 'John Abraham',
    'Juhi Chawla', 'Kajal Aggarwal', 'Kajol', 'Kalki Koechlin', 'Kamal Haasan',
    'Kangana Ranaut', 'Kareena Kapoor', 'Karisma Kapoor', 'Kartik Aaryan', 'Katrina Kaif',
    'Kay Kay Menon', 'Kiara Advani', 'Konkona Sen Sharma', 'Kriti Kharbanda', 'Kriti Sanon',
    'Kunal Khemu', 'Lara Dutta', 'Madhuri Dixit', 'Mahesh Babu', 'Mahira Khan',
    'Manoj Bajpayee', 'Mithun Chakraborty', 'Mohanlal', 'Mrunal Thakur', 'N T Rama Rao Jr',
    'Nagarjuna', 'Nana Patekar', 'Nargis Fakhri', 'Naseeruddin Shah', 'Nawazuddin Siddiqui',
    'Nayanthara', 'Neha Sharma', 'Nimrat Kaur', 'Nithya Menen', 'Nushrat Bharucha',
    'Om Puri', 'Pankaj Kapur', 'Pankaj Tripathi', 'Paresh Rawal', 'Parineeti Chopra',
    'Parvathy Thiruvothu Kottuvatta', 'Piyush Mishra', 'Pooja Hegde', 'Prabhas', 'Prachi Desai',
    'Prateik Babbar', 'Preity Zinta', 'Priyanka Chopra', 'R Madhavan', 'Raashi Khanna',
    'Radhika Apte', 'Rajinikanth', 'Rajkummar Rao', 'Rakul Preet Singh', 'Ram Charan',
    'Rana Daggubati', 'Ranbir Kapoor', 'Randeep Hooda', 'Rani Mukerji', 'Ranveer Singh',
    'Ranvir Shorey', 'Rashmika Mandanna', 'Richa Chadda', 'Rishi Kapoor', 'Riteish Deshmukh',
    'Sai Pallavi', 'Saif Ali Khan', 'Salman Khan', 'Samantha Ruth Prabhu', 'Saniya Malhotra',
    'Sanjay Dutt', 'Sara Ali Khan', 'Shah Rukh Khan', 'Shahid Kapoor', 'Shalini Pandey',
    'Sharman Joshi', 'Shilpa Shetty', 'Shraddha Kapoor', 'Shreyas Talpade', 'Shriya Pilgaonkar',
    'Shriya Saran', 'Shruti Haasan', 'Sidharth Malhotra', 'Soha Ali Khan', 'Sonakshi Sinha',
    'Sonam Kapoor', 'Sonu Sood', 'Sudeep', 'Suniel Shetty', 'Sunny Deol', 'Suriya',
    'Sushant Singh Rajput', 'Taapsee Pannu', 'Tabu', 'Tamannaah Bhatia', 'Tiger Shroff',
    'Trisha Krishnan', 'Tusshar Kapoor', 'Twinkle Khanna', 'Uday Chopra', 'Urvashi Rautela',
    'Vaani Kapoor', 'Varun Dhawan', 'Vicky Kaushal', 'Vidya Balan', 'Vidyut Jamal',
    'Vijay Deverakonda', 'Vijay Raaz', 'Vijay Sethupathi', 'Vikram', 'Vinay Pathak',
    'Vivek Oberoi', 'Yami Gautam', 'Yash', 'Zareen Khan', 'woman', 'a men', 'a man'
]

# Function to load image from URL or file
def load_image(image_path_or_url):
    try:
        if isinstance(image_path_or_url, str) and image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to generate caption
def generate_caption(image, processor, model, device, max_new_tokens=50):
    if image is None:
        return "No image provided."

    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        for actor in indian_actors:
            caption = caption.replace(actor, "")
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "Error generating caption."

@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    model.to(torch.device("cpu"))
    return processor, model

# Main function
def main():
    # Load model and processor
    processor, model = load_model_and_processor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    st.title("Image Captioning App")

    st.header("Input Image")
    uploaded_file = st.file_uploader("Drag and drop or browse for an image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or, enter an image URL")

    if st.button("Generate Description"):
        input_image = None

        # Load image from file uploader
        if uploaded_file is not None:
            input_image = uploaded_file
        # Load image from URL
        elif image_url:
            input_image = image_url

        # Perform caption generation if image is provided
        if input_image is not None:
            image = load_image(input_image)
            if image is not None:
                caption = generate_caption(image, processor, model, device, max_new_tokens=50)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.success(f"Generated Caption: {caption}")
        else:
            st.error("Please provide an image through upload or URL.")

if __name__ == "__main__":
    main()
