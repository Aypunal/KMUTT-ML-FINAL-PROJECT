'use client';

import { useState } from "react";
import { FiUpload, FiX } from "react-icons/fi";

const UploadButton = ({ isUpload }) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedFiles, setSelectedFiles] = useState([]);

    // Handle file selection
    const handleFileSelect = (event) => {
        const files = Array.from(event.target.files);
        setSelectedFiles((prev) => [...prev, ...files]);
    };

    // Handle drag-and-drop functionality
    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const files = Array.from(event.dataTransfer.files);
        setSelectedFiles((prev) => [...prev, ...files]);
    };

    // Remove a file
    const handleRemoveFile = (index) => {
        setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    };

    const handleUpload = async () => {
        if (selectedFiles.length === 0) {
        alert("No files selected!");
        return;
        }

        const formData = new FormData();
        selectedFiles.forEach((file) => {
        formData.append("files", file);
        });
        isUpload(true, selectedFiles);
        setIsModalOpen(false);
        setSelectedFiles([]);
        // try {
        //   // Example API endpoint
        //   const response = await fetch("/api/upload", {
        //     method: "POST",
        //     body: formData,
        //   });

        //   if (!response.ok) throw new Error("Upload failed!");

        //   alert("Files uploaded successfully!");
        //   setSelectedFiles([]); // Clear files after successful upload
        // } catch (error) {
        //   console.error(error);
        //   alert("Error uploading files.");
        // }
    };

    return (
        <>
        {/* Button to Open Modal */}
        <button
            onClick={() => setIsModalOpen(true)}
            className="px-4 py-2 bg-purple-600 hover:bg-black border-2 border-solid cursor-pointer hover:text-purple-600 border-purple-600 text-white w-fit rounded-md transition-all duration-300 ease-in-out"
        >
            Predict Image
        </button>

        {/* Modal */}
        {isModalOpen && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-gray-600 p-6 rounded-lg shadow-lg w-full max-w-lg">
                {/* Modal Header */}
                <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold text-white">Upload Images</h2>
                <button
                    onClick={() => setIsModalOpen(false)}
                    className="text-white "
                >
                    <FiX size={20} />
                </button>
                </div>

                {/* Drag and Drop Area */}
                <div
                className="flex flex-col items-center justify-center border-2 border-dashed border-white rounded-lg bg-gray-600 p-8"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                >
                <FiUpload size={50} className="text-white" />
                <p className="text-white text-sm">Drag and drop images here</p>
                <label
                    htmlFor="file-upload"
                    className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-md cursor-pointer hover:bg-blue-600 transition-all duration-300 ease-in-out"
                >
                    Select Images
                </label>
                    <input
                    id="file-upload"
                    type="file"
                    accept="image/*" // Restricts file types to images
                    multiple
                    onChange={handleFileSelect}
                    className="hidden"
                    />
                </div>

                {/* Selected Files Preview */}
                {selectedFiles.length > 0 && (
                <div className="mt-4">
                    <h3 className="font-bold text-white">Selected Images:</h3>
                    <ul className="mt-2 space-y-2">
                    {selectedFiles.map((file, index) => (
                        <li
                        key={index}
                        className="flex justify-between items-center bg-gray-400 p-2 rounded-md"
                        >
                        <span className="truncate w-4/5 text-white">{file.name}</span>
                        <button
                            onClick={() => handleRemoveFile(index)}
                            className="text-red-500 hover:text-red-700"
                        >
                            <FiX size={18} />
                        </button>
                        </li>
                    ))}
                    </ul>

                    {/* Upload Button */}
                    <button
                    onClick={handleUpload}
                    className="mt-4 bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 w-full"
                    >
                    Upload {selectedFiles.length} Images
                    {selectedFiles.length > 1 ? "s" : ""}
                    </button>
                </div>
                )}
            </div>
            </div>
        )}
        </>
    );
};

export default UploadButton;