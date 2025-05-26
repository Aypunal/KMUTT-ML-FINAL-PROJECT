"use client";
import React from "react";
import { useState } from "react";
import Image from "next/image";
import UploadButton from "./component/uploadButton";
import LoadingScreen from "./component/loadingScreen";
import ScoreSlider from "./component/scoreBar";

export default function Home() {
    const [isUpload, setIsUpload] = useState(false);
    const [images, setImages] = useState([]); // State to hold uploaded images
    const [result, setResult] = useState(null);
    const [predict, setPredict] = useState(false);
    const handleUpload = (status, uploadedImages) => {
      setResult(null);
      setIsUpload(status);
      setImages(uploadedImages); // Save the uploaded images
    }; 
    const handlePredict = async (image) => {
        if (image.length === 0) {
            alert("No images selected!");
            return;
        }
        setPredict(true);
        const formData = new FormData();
        image.forEach((file) => {
            formData.append("image", file);
        });

        try {
            const response = await fetch("https://mlopsweb.onrender.com/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Prediction failed!");

            const data = await response.json();
            setResult(data);   
            alert("Prediction successful!");
            setPredict(false);
        } catch (error) {
            console.error(error);
            alert("Error during prediction.");
        }
    }
    
  return (
    //create a div with that have upload file functionality
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h1 className="text-4xl font-bold mb-4">AI Image Prediction</h1>
      {!isUpload && <UploadButton isUpload={handleUpload}/>}
      {isUpload && (
        <div className="flex flex-col items-center justify-center mt-4">
          <h2 className="text-2xl font-bold mb-4">Uploaded Images</h2>
          <p className="font-bold ">*Score &gt; 50% = <span className="text-red-400">AI-Generated</span>*</p>
          <div className="mt-4 items-center">
            <div className={result ? "grid grid-cols-2 gap-4" : null}>
              <div className="flex items-center justify-center">
                {images.map((image, index) => (
                <Image
                  key={index}
                  src={URL.createObjectURL(image)}
                  alt={`Uploaded Image ${index + 1}`}
                  width={200}
                  height={200}
                  className="rounded-lg shadow-lg"
                />
              ))}
              </div>
              {result ? (<div className="flex flex-col">
                <h1 className="text-xl font-bold">Prediction :</h1>
                <ScoreSlider score={result.prediction}/>
                <p>Prediction Score : {(result.prediction * 100).toFixed(2)}%</p>
              </div>) : null}
            </div>
             {result ? (<div className="flex items-center justify-center mt-4">
              <UploadButton isUpload={handleUpload}/>
             </div>) : (
              <div className="flex items-center justify-center mt-4">
                <button
                onClick={() => handlePredict(images)}
                className="px-4 py-2 bg-purple-600 hover:bg-black border-2 border-solid cursor-pointer hover:text-purple-600 border-purple-600 text-white w-fit rounded-md transition-all duration-300 ease-in-out"
                >
                    Predict
                </button>
              </div>
             )}
          </div>
        </div>
      )}
      {predict && <LoadingScreen />}
    </div>
  );
}
