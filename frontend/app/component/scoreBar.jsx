import { useState, useEffect } from 'react';

export default function ScoreSlider({ score }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    setProgress(0);
    const timeout = setTimeout(() => {
      setProgress(score * 100);
    }, 50);

    return () => clearTimeout(timeout);
  }, [score]);

  return (
    <div className="relative w-full h-8 bg-white border border-gray-300 rounded-full overflow-hidden">
      <div
        className="absolute top-0 left-0 h-full bg-green-500 transition-all duration-1000 rounded-full"
        style={{ width: `${progress}%` }}
      />
      {/* Centered text over the entire bar */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none font-semibold text-gray-800">
        {progress.toFixed(2)}%
      </div>
    </div>
  );
}


