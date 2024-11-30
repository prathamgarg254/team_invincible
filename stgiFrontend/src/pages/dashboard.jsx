import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { AnimatedoneD } from "../components/animated1d";
import bgImage from "../assets/Untitled.png"

export const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [originalImage, setOriginalImage] = useState('');
  const [elaImage, setElaImage] = useState('');

  // Handle file changes from input or drop
  function handleChange(e) {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile); // Store the file itself, not the URL
      const fileURL = URL.createObjectURL(selectedFile);
      // Optionally, if you want to display the image
      setOriginalImage(fileURL);
    }
  }

  // Handle drag and drop events
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      const fileURL = URL.createObjectURL(droppedFile);
      setOriginalImage(fileURL); // Display the image
    }
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  // Trigger file input click when clicking the whole drop zone
  const handleClick = () => {
    fileInputRef.current.click();
  };

  // Function to upload the image to the backend
  const uploadImage = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file); // Append the file to the FormData object

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // Set the content type
        },
      });

      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
      setElaImage(response.data.ela_image_url);
      // If you want to display the original image, set the image here
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  useEffect(() => {
    if (file) {
      uploadImage(); // Call uploadImage whenever a new file is selected
    }
  }, [file]);

  return (
    <div className="" style={{ backgroundImage: `url(${bgImage})` }}>
      <div className="h-lvh grid grid-cols-2 bg-cover bg-center bg-no-repeat" style={{ backgroundImage: `url(${bgImage})` }}>
        <div className="text-green-800 items-center flex flex-col justify-center px-24">
          <div className="font-extrabold text-6xl items-center">
            Tag Trash
          </div>
          <div className="pt-8 text-xl">
          Tagging waste intelligently for a cleaner world.
          </div>
          <div className="text-center text-xl">
            <img src="./src/assets/rec" alt="" />
          </div>
        </div>
        <div className="flex flex-col justify-center items-center m-14">
          <AnimatedoneD
            content={
              <>
                <div className="text-center mb-14 text-xl">
                  Drag and drop the image you want to check
                </div>
                <div
                  className={`border-2 border-dashed p-8 rounded-lg mb-4 mr-4 w-full h-64 flex justify-center items-center cursor-pointer ${
                    isDragging ? "border-blue-500 bg-blue-100" : "border-gray-400"
                  }`}
                  onClick={handleClick}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onDragLeave={handleDragLeave}
                >
                  {isDragging ? (
                    <p>Drop your file here...</p>
                  ) : (
                    <p>Drag or drop your file here, or click to select</p>
                  )}
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleChange}
                    className="hidden"
                  />
                </div>
                {originalImage && (
                  <img src={originalImage} alt="Uploaded File" className="w-[200px] h-[200px]" />
                )}
              </>
            }
          />
        </div>
      </div>
      <div className="items-center flex justify-center text-3xl font-medium">Results</div>
      <div className="h-auto min-w-full flex justify-around">
      
        
        {/* Display the original image */}
        {originalImage && (
          <div className="flex flex-col items-center mt-4">
            <h3 className="text-xl font-bold">Original Image</h3>
            <img src={originalImage} alt="Original Result" className="w-[400px] h-[400px]" />
          </div>
        )}
        
        {/* Display the ELA image */}
        {elaImage && (
          <div className="flex flex-col items-center mt-4">
            <h3 className="text-xl font-bold">ELA Image</h3>
            <img src={elaImage} alt="ELA Result" className="w-[400px] h-[400px]" />
          </div>
        )}
        
        
      </div>
      <div className="flex flex-col items-center mt-4">
          <h3 className="text-xl">Prediction: {prediction}</h3>
          <h3 className="text-xl">Confidence: {confidence}</h3>
        </div>
      {/* <div className="h-[1200px] px-10 py-8">
        <div className="items-center flex justify-center text-3xl font-medium">Developers</div>
        <div className="grid grid-cols-3 h-[600px] px-10 py-8">
          <div className="mx-8 my-2">
            <AnimatedoneD 
              content={
                <>
                  <div className="text-center text-xl">
                  <img 
                      src="./src/assets/WhatsApp Image 2024-09-29 at 04.17.54.jpeg"  // Replace with the actual path to your image
                      alt=""
                      className="rounded-full w-[350px] h-[400px] mx-auto mb-4" // Adjust classes as needed
                  />
                    <div className="text-3xl mt-30 font-semibold">Gauravpreet Singh Padda</div>
                  </div>
                </>
              }
            />
          </div>
          <div className="mx-8 my-2">
            <AnimatedoneD 
              content={
                <>
                  <div className="text-center text-xl">
                  <img 
                      src="./src/assets/WhatsApp Image 2024-09-29 at 04.31.54.jpeg"  // Replace with the actual path to your image
                      alt=""
                      className="rounded-full w-[350px] h-[400px] mx-auto mb-4" // Adjust classes as needed
                  />
                    <div className="text-3xl font-semibold">Vishwas Kisaniya</div>
                  </div>
                </>
              }
            />
          </div>
          <div className="mx-8 my-2">
            <AnimatedoneD 
              content={
                <>
                  <div className="text-center text-xl">
                  <img 
                      src="./src/assets/WhatsApp Image 2024-09-29 at 04.17.53.jpeg"  // Replace with the actual path to your image
                      alt=""
                      className="rounded-full w-[300px] h-[400px] mx-auto mb-4" // Adjust classes as needed
                  />
                    <div className="text-3xl font-semibold">Keshav Garg</div>
                  </div>
                </>
              }
            />
          </div>
          <div className="mx-8 my-2">
            <AnimatedoneD 
              content={
                <>
                  <div className="text-center text-xl">
                  <img 
                      src="./src/assets/WhatsApp Image 2024-09-29 at 04.35.06.jpeg"  // Replace with the actual path to your image
                      alt=""
                      className="rounded-full w-[350px] h-[400px] mx-auto mb-4" // Adjust classes as needed
                  />
                    <div className="text-3xl font-semibold">Jashanpreet Singh Salh</div>
                  </div>
                </>
              } 
            />
          </div>
          <div className="mx-8 my-2">
            <AnimatedoneD 
              content={
                <>
                  <div className="text-center text-xl">
                  <img 
                      src="./src/assets/pr.jpeg"  // Replace with the actual path to your image
                      alt="Pranshu Chauhan"
                      className="rounded-full w-[350px] h-[400px] mx-auto mb-4" // Adjust classes as needed
                  />
                    <div className="text-3xl font-semibold">Pranshu Chauhan</div>
                  </div>
                </>
              }  
            />
          </div>
        </div>
      </div>

      <div className="flex flex-col justify-center items-center h-64 px-10 py-8 bg-slate-200">
        <div className="text-3xl font-semibold p-2">Made by Team Invincibles</div>
        <div className="text-xl p-2 font-thin ">
          "We hope you enjoyed exploring our project! Your feedback is invaluable as we continue improving our image authentication solution. Thank you for your support!"
        </div>
      </div> */}
    </div>
  );
};
