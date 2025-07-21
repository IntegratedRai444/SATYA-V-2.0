import * as React from "react"
import * as SeparatorPrimitive from "@radix-ui/react-separator"

<<<<<<< HEAD
import { cn } from "../../lib/utils";
=======
import { cn } from "@/lib/utils"
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>
>(
  (
    { className, orientation = "horizontal", decorative = true, ...props },
    ref
  ) => (
    <SeparatorPrimitive.Root
      ref={ref}
      decorative={decorative}
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border",
        orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]",
        className
      )}
      {...props}
    />
  )
)
Separator.displayName = SeparatorPrimitive.Root.displayName

export { Separator }
