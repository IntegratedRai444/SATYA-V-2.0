<<<<<<< HEAD
import React from 'react';
import { cn } from "../../lib/utils";
=======
import { cn } from "@/lib/utils"
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }
