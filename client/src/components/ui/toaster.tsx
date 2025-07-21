<<<<<<< HEAD
import React from 'react';
import { useToast } from "../../hooks/use-toast";
=======
import { useToast } from "@/hooks/use-toast"
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
<<<<<<< HEAD
} from "./toast";
=======
} from "@/components/ui/toast"
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f

export function Toaster() {
  const { toasts } = useToast()

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, ...props }) {
        return (
          <Toast key={id} {...props}>
            <div className="grid gap-1">
              {title && <ToastTitle>{title}</ToastTitle>}
              {description && (
                <ToastDescription>{description}</ToastDescription>
              )}
            </div>
            {action}
            <ToastClose />
          </Toast>
        )
      })}
      <ToastViewport />
    </ToastProvider>
  )
}
