import { useState, useCallback } from 'react';

export interface FileValidationConfig {
    validTypes: string[];
    maxSize: number;
    errorMessage: string;
}

export function useFileUpload(validationConfig: FileValidationConfig) {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [error, setError] = useState<string>('');

    const validateFile = useCallback((file: File): boolean => {
        if (!validationConfig.validTypes.includes(file.type)) {
            setError(validationConfig.errorMessage);
            return false;
        }

        if (file.size > validationConfig.maxSize) {
            const maxSizeMB = validationConfig.maxSize / (1024 * 1024);
            setError(`File size must be less than ${maxSizeMB}MB`);
            return false;
        }

        setError('');
        return true;
    }, [validationConfig]);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent, onFileSelect?: (file: File) => void) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (validateFile(file)) {
                setSelectedFile(file);
                onFileSelect?.(file);
            }
        }
    }, [validateFile]);

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>, onFileSelect?: (file: File) => void) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            if (validateFile(file)) {
                setSelectedFile(file);
                onFileSelect?.(file);
            }
        }
    }, [validateFile]);

    const resetFile = useCallback(() => {
        setSelectedFile(null);
        setError('');
    }, []);

    return {
        dragActive,
        selectedFile,
        error,
        setError,
        handleDrag,
        handleDrop,
        handleFileChange,
        resetFile,
        setSelectedFile,
    };
}
