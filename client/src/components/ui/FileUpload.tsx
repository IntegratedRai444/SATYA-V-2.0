import React, { useCallback, useState } from 'react';
import { FiUpload, FiFile, FiX, FiImage, FiVideo, FiFileText } from 'react-icons/fi';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  maxSizeMB?: number;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFilesSelected,
  accept = 'image/*,video/*,.pdf,.doc,.docx',
  multiple = true,
  maxSizeMB = 50,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return <FiImage className="text-purple-400" />;
    if (file.type.startsWith('video/')) return <FiVideo className="text-purple-400" />;
    return <FiFileText className="text-purple-400" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateFiles = (files: FileList): File[] => {
    const validFiles: File[] = [];
    const maxSize = maxSizeMB * 1024 * 1024; // Convert MB to bytes

    Array.from(files).forEach((file) => {
      if (file.size > maxSize) {
        setError(`File ${file.name} exceeds the maximum size of ${maxSizeMB}MB`);
        return;
      }
      validFiles.push(file);
    });

    return validFiles;
  };

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const validFiles = validateFiles(files);
        if (validFiles.length > 0) {
          setSelectedFiles((prev) => [...prev, ...validFiles]);
          onFilesSelected(validFiles);
          setError(null);
        }
      }
    },
    [onFilesSelected, maxSizeMB]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const validFiles = validateFiles(files);
      if (validFiles.length > 0) {
        setSelectedFiles((prev) => [...prev, ...validFiles]);
        onFilesSelected(validFiles);
        setError(null);
      }
    }
  };

  const removeFile = (index: number) => {
    const newFiles = [...selectedFiles];
    newFiles.splice(index, 1);
    setSelectedFiles(newFiles);
    onFilesSelected(newFiles);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  return (
    <div className="w-full">
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          isDragging ? 'border-purple-500 bg-purple-900/20' : 'border-gray-700 hover:border-purple-500/50'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="flex flex-col items-center justify-center">
          <div className="w-12 h-12 rounded-full bg-purple-900/50 flex items-center justify-center mb-4">
            <FiUpload className="text-xl text-purple-400" />
          </div>
          <h3 className="text-lg font-medium text-white mb-1">Drop files here</h3>
          <p className="text-gray-400 text-sm mb-4">
            or{' '}
            <label className="text-purple-400 hover:text-purple-300 cursor-pointer">
              browse files
              <input
                type="file"
                className="hidden"
                onChange={handleFileInput}
                accept={accept}
                multiple={multiple}
              />
            </label>
          </p>
          <p className="text-xs text-gray-500">
            Supports: Images, Videos, PDFs, Word (max {maxSizeMB}MB)
          </p>
        </div>
      </div>

      {error && <p className="mt-2 text-sm text-red-400">{error}</p>}

      {selectedFiles.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-sm font-medium text-gray-300 mb-2">Selected Files</h4>
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between bg-gray-800/50 rounded-lg p-3"
              >
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-gray-700/50 rounded-lg">
                    {getFileIcon(file)}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-200 truncate max-w-xs">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-white p-1 rounded-full hover:bg-gray-700"
                >
                  <FiX className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
