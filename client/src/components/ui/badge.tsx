import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-accent-cyan focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-accent-cyan text-white hover:bg-accent-cyan-dark",
        secondary:
          "border-transparent bg-bg-tertiary text-text-secondary hover:bg-bg-tertiary/80",
        destructive:
          "border-transparent bg-accent-red text-white hover:bg-accent-red/80",
        outline: "text-text-primary border-border-primary",
        success:
          "border-transparent bg-accent-green/10 text-accent-green border-accent-green/30",
        warning:
          "border-transparent bg-accent-orange/10 text-accent-orange border-accent-orange/30",
        info:
          "border-transparent bg-accent-cyan/10 text-accent-cyan border-accent-cyan/30",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }