"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getPasswordErrorMessage = exports.validatePassword = void 0;
const validatePassword = (password) => {
    const errors = [];
    if (password.length < 8) {
        errors.push("Password must be at least 8 characters long");
    }
    if (!/[A-Z]/.test(password)) {
        errors.push("Password must contain at least one uppercase letter");
    }
    if (!/[a-z]/.test(password)) {
        errors.push("Password must contain at least one lowercase letter");
    }
    if (!/[0-9]/.test(password)) {
        errors.push("Password must contain at least one number");
    }
    // Optional: Check for special characters
    // if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    //   errors.push("Password must contain at least one special character");
    // }
    return {
        isValid: errors.length === 0,
        errors
    };
};
exports.validatePassword = validatePassword;
// Frontend helper to get a user-friendly error message
const getPasswordErrorMessage = (errors) => {
    if (errors.length === 0)
        return '';
    return errors.join("\n");
};
exports.getPasswordErrorMessage = getPasswordErrorMessage;
