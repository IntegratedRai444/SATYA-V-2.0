import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-accent-cyan text-white hover:bg-accent-cyan-dark shadow-lg shadow-accent-cyan/30 hover:shadow-accent-cyan/50",
        primary: "bg-accent-cyan text-white hover:bg-accent-cyan-dark shadow-lg shadow-accent-cyan/30 hover:shadow-accent-cyan/50",
        secondary: "bg-transparent border-2 border-accent-cyan text-accent-cyan hover:bg-accent-cyan/10",
        outline: "border border-accent-cyan text-text-primary hover:bg-bg-tertiary",
        ghost: "hover:bg-bg-tertiary text-text-secondary hover:text-text-primary",
        gradient: "bg-gradient-to-r from-accent-cyan to-accent-purple text-white shadow-lg",
        destructive: "bg-accent-red text-white hover:bg-accent-red/90",
        link: "text-accent-cyan underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 px-3 text-sm",
        lg: "h-12 px-6 py-3 text-lg",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

// eslint-disable-next-line react-refresh/only-export-components
export { Button, buttonVariants }