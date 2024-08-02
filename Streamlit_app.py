{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/CreatorGeetansh/Streamlit_shoppin/blob/main/Streamlit_app_py.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit\n",
        "!pip install pyngrok\n",
        "!pip install timm einops flash_attn"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l7mxmot6Jdow",
        "outputId": "be59237c-4203-4aaf-b971-e97892aa4a26"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.37.0)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.4.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.1.4)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: watchdog<5,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.0.1)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.10/dist-packages (7.2.0)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.1)\n",
            "Requirement already satisfied: timm in /usr/local/lib/python3.10/dist-packages (1.0.8)\n",
            "Requirement already satisfied: einops in /usr/local/lib/python3.10/dist-packages (0.8.0)\n",
            "Requirement already satisfied: flash_attn in /usr/local/lib/python3.10/dist-packages (2.6.3)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from timm) (2.3.1+cu121)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (from timm) (0.18.1+cu121)\n",
            "Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from timm) (6.0.1)\n",
            "Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.10/dist-packages (from timm) (0.23.5)\n",
            "Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from timm) (0.4.3)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (3.15.4)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (2024.6.1)\n",
            "Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (24.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (2.31.0)\n",
            "Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (4.66.4)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (4.12.2)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->timm) (1.13.1)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->timm) (3.3)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (3.1.4)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (8.9.2.26)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.3.1)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (11.0.2.54)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (10.3.2.106)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (11.4.5.107)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.0.106)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (2.20.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)\n",
            "Requirement already satisfied: triton==2.3.1 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (2.3.1)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch->timm) (12.6.20)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (1.26.4)\n",
            "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (9.4.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->timm) (2.1.5)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (2024.7.4)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->timm) (1.3.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YRFXktmcRvWq",
        "outputId": "e6d46499-7cdb-441d-ceea-9d1adf34eaef"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "from PIL import Image\n",
        "from transformers import AutoProcessor, AutoModelForCausalLM\n",
        "import requests\n",
        "from io import BytesIO\n",
        "import torch\n",
        "\n",
        "indian_actors = [\n",
        "    'Aamir Khan', 'Abhay Deol', 'Abhishek Bachchan', 'Adah Sharma', 'Aditi Rao Hydari',\n",
        "    'Aditya Roy Kapoor', 'Aftab Shivdasani', 'Aishwarya Rai', 'Ajay Devgn', 'Akshay Kumar',\n",
        "    'Akshaye Khanna', 'Alia Bhatt', 'Allu Arjun', 'Ameesha Patel', 'Amitabh Bachchan',\n",
        "    'Amrish Puri', 'Amrita Rao', 'Amy Jackson', 'Ananya Panday', 'Anil Kapoor',\n",
        "    'Anupam Kher', 'Anushka Sharma', 'Anushka Shetty', 'Arjun Kapoor', 'Arjun Rampal',\n",
        "    'Arshad Warsi', 'Asin', 'Ayesha Takia', 'Ayushmann Khurrana', 'Bhumi Pednekar',\n",
        "    'Bipasha Basu', 'Bobby Deol', 'Boman Irani', 'Chiranjeevi', 'Chitrangda Singh',\n",
        "    'Chunky Pandey', 'Deepika Padukone', 'Dhanush', 'Dia Mirza', 'Disha Patani',\n",
        "    'Dulquer Salmaan', 'Emraan Hashmi', 'Esha Gupta', 'Farhan Akhtar', 'Fatima Sana Shaikh',\n",
        "    'Govinda', 'Gul Panag', 'Hrithik Roshan', 'Huma Qureshi', 'Ileana DCruz',\n",
        "    'Irrfan Khan', 'Jacqueline Fernandez', 'Janvi Kapoor', 'Jimmy Shergill', 'John Abraham',\n",
        "    'Juhi Chawla', 'Kajal Aggarwal', 'Kajol', 'Kalki Koechlin', 'Kamal Haasan',\n",
        "    'Kangana Ranaut', 'Kareena Kapoor', 'Karisma Kapoor', 'Kartik Aaryan', 'Katrina Kaif',\n",
        "    'Kay Kay Menon', 'Kiara Advani', 'Konkona Sen Sharma', 'Kriti Kharbanda', 'Kriti Sanon',\n",
        "    'Kunal Khemu', 'Lara Dutta', 'Madhuri Dixit', 'Mahesh Babu', 'Mahira Khan',\n",
        "    'Manoj Bajpayee', 'Mithun Chakraborty', 'Mohanlal', 'Mrunal Thakur', 'N T Rama Rao Jr',\n",
        "    'Nagarjuna', 'Nana Patekar', 'Nargis Fakhri', 'Naseeruddin Shah', 'Nawazuddin Siddiqui',\n",
        "    'Nayanthara', 'Neha Sharma', 'Nimrat Kaur', 'Nithya Menen', 'Nushrat Bharucha',\n",
        "    'Om Puri', 'Pankaj Kapur', 'Pankaj Tripathi', 'Paresh Rawal', 'Parineeti Chopra',\n",
        "    'Parvathy Thiruvothu Kottuvatta', 'Piyush Mishra', 'Pooja Hegde', 'Prabhas', 'Prachi Desai',\n",
        "    'Prateik Babbar', 'Preity Zinta', 'Priyanka Chopra', 'R Madhavan', 'Raashi Khanna',\n",
        "    'Radhika Apte', 'Rajinikanth', 'Rajkummar Rao', 'Rakul Preet Singh', 'Ram Charan',\n",
        "    'Rana Daggubati', 'Ranbir Kapoor', 'Randeep Hooda', 'Rani Mukerji', 'Ranveer Singh',\n",
        "    'Ranvir Shorey', 'Rashmika Mandanna', 'Richa Chadda', 'Rishi Kapoor', 'Riteish Deshmukh',\n",
        "    'Sai Pallavi', 'Saif Ali Khan', 'Salman Khan', 'Samantha Ruth Prabhu', 'Saniya Malhotra',\n",
        "    'Sanjay Dutt', 'Sara Ali Khan', 'Shah Rukh Khan', 'Shahid Kapoor', 'Shalini Pandey',\n",
        "    'Sharman Joshi', 'Shilpa Shetty', 'Shraddha Kapoor', 'Shreyas Talpade', 'Shriya Pilgaonkar',\n",
        "    'Shriya Saran', 'Shruti Haasan', 'Sidharth Malhotra', 'Soha Ali Khan', 'Sonakshi Sinha',\n",
        "    'Sonam Kapoor', 'Sonu Sood', 'Sudeep', 'Suniel Shetty', 'Sunny Deol', 'Suriya',\n",
        "    'Sushant Singh Rajput', 'Taapsee Pannu', 'Tabu', 'Tamannaah Bhatia', 'Tiger Shroff',\n",
        "    'Trisha Krishnan', 'Tusshar Kapoor', 'Twinkle Khanna', 'Uday Chopra', 'Urvashi Rautela',\n",
        "    'Vaani Kapoor', 'Varun Dhawan', 'Vicky Kaushal', 'Vidya Balan', 'Vidyut Jamal',\n",
        "    'Vijay Deverakonda', 'Vijay Raaz', 'Vijay Sethupathi', 'Vikram', 'Vinay Pathak',\n",
        "    'Vivek Oberoi', 'Yami Gautam', 'Yash', 'Zareen Khan', 'woman', 'a men', 'a man'\n",
        "]\n",
        "\n",
        "# Function to load image from URL or file\n",
        "def load_image(image_path_or_url):\n",
        "    try:\n",
        "        if isinstance(image_path_or_url, str) and image_path_or_url.startswith(('http://', 'https://')):\n",
        "            response = requests.get(image_path_or_url)\n",
        "            response.raise_for_status()\n",
        "            image = Image.open(BytesIO(response.content)).convert(\"RGB\")\n",
        "        else:\n",
        "            image = Image.open(image_path_or_url).convert(\"RGB\")\n",
        "        return image\n",
        "    except Exception as e:\n",
        "        st.error(f\"Error loading image: {e}\")\n",
        "        return None\n",
        "\n",
        "# Function to generate caption\n",
        "def generate_caption(image, processor, model, device, max_new_tokens=50):\n",
        "    if image is None:\n",
        "        return \"No image provided.\"\n",
        "\n",
        "    try:\n",
        "        inputs = processor(images=image, return_tensors=\"pt\").to(device)\n",
        "        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)\n",
        "        caption = processor.decode(outputs[0], skip_special_tokens=True)\n",
        "        for actor in indian_actors:\n",
        "            caption = caption.replace(actor, \"\")\n",
        "        return caption\n",
        "    except Exception as e:\n",
        "        st.error(f\"Error generating caption: {e}\")\n",
        "        return \"Error generating caption.\"\n",
        "\n",
        "@st.cache_resource\n",
        "def load_model_and_processor():\n",
        "    processor = AutoProcessor.from_pretrained(\"microsoft/Florence-2-base\", trust_remote_code=True)\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"microsoft/Florence-2-large\", trust_remote_code=True)\n",
        "    model.to(torch.device(\"cpu\"))\n",
        "    return processor, model\n",
        "\n",
        "# Main function\n",
        "def main():\n",
        "    # Load model and processor\n",
        "    processor, model = load_model_and_processor()\n",
        "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "    model.to(device)\n",
        "\n",
        "    st.title(\"Image Captioning App\")\n",
        "\n",
        "    st.header(\"Input Image\")\n",
        "    uploaded_file = st.file_uploader(\"Drag and drop or browse for an image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
        "    image_url = st.text_input(\"Or, enter an image URL\")\n",
        "\n",
        "    if st.button(\"Generate Description\"):\n",
        "        input_image = None\n",
        "\n",
        "        # Load image from file uploader\n",
        "        if uploaded_file is not None:\n",
        "            input_image = uploaded_file\n",
        "        # Load image from URL\n",
        "        elif image_url:\n",
        "            input_image = image_url\n",
        "\n",
        "        # Perform caption generation if image is provided\n",
        "        if input_image is not None:\n",
        "            image = load_image(input_image)\n",
        "            if image is not None:\n",
        "                caption = generate_caption(image, processor, model, device, max_new_tokens=50)\n",
        "                st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
        "                st.success(f\"Generated Caption: {caption}\")\n",
        "        else:\n",
        "            st.error(\"Please provide an image through upload or URL.\")\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 411
        },
        "id": "Jkg2owTgSYxM",
        "outputId": "317613f7-ef90-4f8e-9f95-f91193328ae9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ngrok URL: NgrokTunnel: \"https://7006-34-122-138-224.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:pyngrok.process.ngrok:t=2024-07-31T19:14:19+0000 lvl=warn msg=\"Stopping forwarder\" name=http-8501-daad6bc0-a2ed-4f77-a3e2-a7b518dc3445 acceptErr=\"failed to accept connection: Listener closed\"\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-11-085b8990eee0>\u001b[0m in \u001b[0;36m<cell line: 19>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0;31m# Run the Streamlit app and ngrok tunnel concurrently\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mconcurrent\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfutures\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mThreadPoolExecutor\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 19\u001b[0;31m \u001b[0;32mwith\u001b[0m \u001b[0mThreadPoolExecutor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mexecutor\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     20\u001b[0m     \u001b[0mexecutor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msubmit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrun_streamlit\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     21\u001b[0m     \u001b[0mexecutor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msubmit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstart_ngrok\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.10/concurrent/futures/_base.py\u001b[0m in \u001b[0;36m__exit__\u001b[0;34m(self, exc_type, exc_val, exc_tb)\u001b[0m\n\u001b[1;32m    647\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    648\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__exit__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexc_type\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexc_val\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexc_tb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 649\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshutdown\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mwait\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    650\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    651\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.10/concurrent/futures/thread.py\u001b[0m in \u001b[0;36mshutdown\u001b[0;34m(self, wait, cancel_futures)\u001b[0m\n\u001b[1;32m    233\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mwait\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    234\u001b[0m             \u001b[0;32mfor\u001b[0m \u001b[0mt\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_threads\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 235\u001b[0;31m                 \u001b[0mt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    236\u001b[0m     \u001b[0mshutdown\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__doc__\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_base\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mExecutor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshutdown\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__doc__\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.10/threading.py\u001b[0m in \u001b[0;36mjoin\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m   1094\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1095\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mtimeout\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1096\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_wait_for_tstate_lock\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1097\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1098\u001b[0m             \u001b[0;31m# the behavior of a negative timeout isn't documented, but\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.10/threading.py\u001b[0m in \u001b[0;36m_wait_for_tstate_lock\u001b[0;34m(self, block, timeout)\u001b[0m\n\u001b[1;32m   1114\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1115\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1116\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mlock\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0macquire\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mblock\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1117\u001b[0m                 \u001b[0mlock\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrelease\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1118\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_stop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "# from pyngrok import ngrok\n",
        "# ngrok.set_auth_token('2dRIwMKJRYDo81avUTWE0dce3Z4_4GNBg9puQGdRydxUFvRfX')\n",
        "# import subprocess\n",
        "\n",
        "# import subprocess\n",
        "# from pyngrok import ngrok, conf\n",
        "\n",
        "# # Function to run Streamlit app\n",
        "# def run_streamlit():\n",
        "#     subprocess.run([\"streamlit\", \"run\", \"app.py\"])\n",
        "\n",
        "# # Function to start ngrok tunnel and print the URL\n",
        "# def start_ngrok():\n",
        "#     url = ngrok.connect(8501)\n",
        "#     print(f\"ngrok URL: {url}\")\n",
        "\n",
        "# # Run the Streamlit app and ngrok tunnel concurrently\n",
        "# from concurrent.futures import ThreadPoolExecutor\n",
        "# with ThreadPoolExecutor() as executor:\n",
        "#     executor.submit(run_streamlit)\n",
        "#     executor.submit(start_ngrok)"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNl5q7t2h+a9RTuYcylCxLo",
      "include_colab_link": false
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
