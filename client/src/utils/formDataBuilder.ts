/**
 * FormData Builder Utility
 * Eliminates FormData building duplication across:
 * - chatService.ts
 * - useBatchProcessing.ts
 * - useMediaAnalysis.ts
 * - Plus backend routes
 */

export interface FormDataOptions {
    onProgress?: (progress: number) => void;
    headers?: Record<string, string>;
}

export class FormDataBuilder {
    private formData: FormData;

    constructor() {
        this.formData = new FormData();
    }

    /**
     * Add a file to the form data
     */
    addFile(fieldName: string, file: File): this {
        this.formData.append(fieldName, file);
        return this;
    }

    /**
     * Add multiple files to the form data
     */
    addFiles(fieldName: string, files: File[]): this {
        files.forEach((file, index) => {
            this.formData.append(`${fieldName}${index}`, file);
        });
        return this;
    }

    /**
     * Add a field to the form data
     */
    addField(fieldName: string, value: string | Blob): this {
        this.formData.append(fieldName, value);
        return this;
    }

    /**
     * Add an object as JSON string
     */
    addJSON(fieldName: string, data: any): this {
        this.formData.append(fieldName, JSON.stringify(data));
        return this;
    }

    /**
     * Add multiple fields from an object
     */
    addFields(fields: Record<string, string | Blob>): this {
        Object.entries(fields).forEach(([key, value]) => {
            this.formData.append(key, value);
        });
        return this;
    }

    /**
     * Get the built FormData
     */
    build(): FormData {
        return this.formData;
    }

    /**
     * Static helper to create FormData from file and options
     */
    static fromFile(file: File, fieldName: string = 'file', options?: Record<string, any>): FormData {
        const builder = new FormDataBuilder();
        builder.addFile(fieldName, file);

        if (options) {
            builder.addJSON('options', options);
        }

        return builder.build();
    }

    /**
     * Static helper to create FormData from multiple files
     */
    static fromFiles(files: File[], fieldName: string = 'files', options?: Record<string, any>): FormData {
        const builder = new FormDataBuilder();
        builder.addFiles(fieldName, files);

        if (options) {
            builder.addJSON('options', options);
        }

        return builder.build();
    }

    /**
     * Static helper to create FormData from object
     */
    static fromObject(data: Record<string, any>): FormData {
        const builder = new FormDataBuilder();

        Object.entries(data).forEach(([key, value]) => {
            if (value instanceof File) {
                builder.addFile(key, value);
            } else if (Array.isArray(value) && value[0] instanceof File) {
                builder.addFiles(key, value);
            } else if (typeof value === 'object') {
                builder.addJSON(key, value);
            } else {
                builder.addField(key, String(value));
            }
        });

        return builder.build();
    }
}

/**
 * Helper function to create FormData with progress tracking
 */
export function createFormDataWithProgress(
    file: File,
    fieldName: string = 'file',
    options?: Record<string, any>
): { formData: FormData; config: any } {
    const formData = FormDataBuilder.fromFile(file, fieldName, options);

    const config = {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent: any) => {
            if (progressEvent.total) {
                const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                // Progress can be tracked via callback
                return progress;
            }
        },
    };

    return { formData, config };
}

// Export convenience function
export const formData = {
    fromFile: FormDataBuilder.fromFile.bind(FormDataBuilder),
    fromFiles: FormDataBuilder.fromFiles.bind(FormDataBuilder),
    fromObject: FormDataBuilder.fromObject.bind(FormDataBuilder),
    withProgress: createFormDataWithProgress,
};
