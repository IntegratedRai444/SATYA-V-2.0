"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

function FloatingPaths({ position }: { position: number }) {
    const paths = Array.from({ length: 36 }, (_, i) => ({
        id: i,
        d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
            380 - i * 5 * position
        } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
            152 - i * 5 * position
        } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
            684 - i * 5 * position
        } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
        color: `rgba(15,23,42,${0.1 + i * 0.03})`,
        width: 0.5 + i * 0.03,
        duration: 20 + (i * 0.3), // Use deterministic duration based on index
    }));

    return (
        <div className="absolute inset-0 pointer-events-none">
            <svg
                className="w-full h-full text-slate-950 dark:text-white"
                viewBox="0 0 696 316"
                fill="none"
            >
                <title>Background Paths</title>
                {paths.map((path) => (
                    <motion.path
                        key={path.id}
                        d={path.d}
                        stroke="currentColor"
                        strokeWidth={path.width}
                        strokeOpacity={0.1 + path.id * 0.03}
                        initial={{ pathLength: 0.3, opacity: 0.6 }}
                        animate={{
                            pathLength: 1,
                            opacity: [0.3, 0.6, 0.3],
                            pathOffset: [0, 1, 0],
                        }}
                        transition={{
                            duration: path.duration,
                            repeat: Number.POSITIVE_INFINITY,
                            ease: "linear",
                        }}
                    />
                ))}
            </svg>
        </div>
    );
}

export function BackgroundPaths({
    title = "Background Paths",
}: {
    title?: string;
}) {
    const words = title.split(" ");

    return (
        <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-white dark:bg-neutral-950">
            <div className="absolute inset-0">
                <FloatingPaths position={1} />
                <FloatingPaths position={-1} />
            </div>

            <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 2 }}
                    className="max-w-4xl mx-auto"
                >
                    <h1 className="text-5xl sm:text-7xl md:text-8xl font-bold mb-8 tracking-tighter">
                        {words.map((word, wordIndex) => (
                            <span
                                key={wordIndex}
                                className="inline-block mr-4 last:mr-0"
                            >
                                {word.split("").map((letter, letterIndex) => (
                                    <motion.span
                                        key={`${wordIndex}-${letterIndex}`}
                                        initial={{ y: 100, opacity: 0 }}
                                        animate={{ y: 0, opacity: 1 }}
                                        transition={{
                                            delay:
                                                wordIndex * 0.1 +
                                                letterIndex * 0.03,
                                            type: "spring",
                                            stiffness: 150,
                                            damping: 25,
                                        }}
                                        className="inline-block text-transparent bg-clip-text 
                                        bg-gradient-to-r from-neutral-900 to-neutral-700/80 
                                        dark:from-white dark:to-white/80"
                                    >
                                        {letter}
                                    </motion.span>
                                ))}
                            </span>
                        ))}
                    </h1>

                    <div
                        className="inline-block group relative bg-gradient-to-b from-black/10 to-white/10 
                        dark:from-white/10 dark:to-black/10 p-px rounded-2xl backdrop-blur-lg 
                        overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300"
                    >
                        <Button
                            variant="ghost"
                            className="rounded-[1.15rem] px-8 py-6 text-lg font-semibold backdrop-blur-md 
                            bg-white/95 hover:bg-white/100 dark:bg-black/95 dark:hover:bg-black/100 
                            text-black dark:text-white transition-all duration-300 
                            group-hover:-translate-y-0.5 border border-black/10 dark:border-white/10
                            hover:shadow-md dark:hover:shadow-neutral-800/50"
                        >
                            <span className="opacity-90 group-hover:opacity-100 transition-opacity">
                                Discover Excellence
                            </span>
                            <span
                                className="ml-3 opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 
                                transition-all duration-300"
                            >
                                →
                            </span>
                        </Button>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}

// Dashboard-specific background without hero content
export function DashboardBackground() {
    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {/* Animated background paths */}
            <div className="absolute inset-0 opacity-20">
                <FloatingPaths position={1} />
                <FloatingPaths position={-1} />
            </div>
            
            {/* Enhanced star field */}
            <div className="absolute inset-0">
                {/* Large glowing stars */}
                <div className="absolute top-10 left-20 w-3 h-3 bg-cyan-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/50" style={{ animationDelay: '0s', animationDuration: '4s' }}></div>
                <div className="absolute top-32 right-16 w-2 h-2 bg-blue-400 rounded-full animate-pulse shadow-lg shadow-blue-400/40" style={{ animationDelay: '1.5s', animationDuration: '3.5s' }}></div>
                <div className="absolute top-16 left-1/3 w-2.5 h-2.5 bg-cyan-300 rounded-full animate-pulse shadow-lg shadow-cyan-300/30" style={{ animationDelay: '2.5s', animationDuration: '5s' }}></div>
                <div className="absolute top-48 right-32 w-1.5 h-1.5 bg-blue-300 rounded-full animate-pulse shadow-md shadow-blue-300/30" style={{ animationDelay: '3.2s', animationDuration: '4.2s' }}></div>
                
                {/* Medium animated stars */}
                <div className="absolute top-24 left-40 w-1 h-1 bg-white rounded-full animate-pulse opacity-80" style={{ animationDelay: '0.8s', animationDuration: '2.8s' }}></div>
                <div className="absolute top-36 right-48 w-1.5 h-1.5 bg-cyan-200 rounded-full animate-pulse opacity-70" style={{ animationDelay: '1.2s', animationDuration: '3.1s' }}></div>
                <div className="absolute top-8 left-60 w-1 h-1 bg-blue-200 rounded-full animate-pulse opacity-60" style={{ animationDelay: '2.1s', animationDuration: '2.9s' }}></div>
                <div className="absolute top-44 right-24 w-0.5 h-0.5 bg-white rounded-full animate-pulse opacity-90" style={{ animationDelay: '0.5s', animationDuration: '2.3s' }}></div>
                
                {/* Small twinkling stars */}
                <div className="absolute top-12 right-64 w-0.5 h-0.5 bg-cyan-100 rounded-full animate-pulse opacity-50" style={{ animationDelay: '1.8s', animationDuration: '1.9s' }}></div>
                <div className="absolute top-28 left-24 w-0.5 h-0.5 bg-blue-100 rounded-full animate-pulse opacity-40" style={{ animationDelay: '0.3s', animationDuration: '2.1s' }}></div>
                <div className="absolute top-52 right-40 w-0.5 h-0.5 bg-white rounded-full animate-pulse opacity-60" style={{ animationDelay: '2.8s', animationDuration: '1.7s' }}></div>
                <div className="absolute top-20 left-80 w-0.5 h-0.5 bg-cyan-50 rounded-full animate-pulse opacity-30" style={{ animationDelay: '1.5s', animationDuration: '2.5s' }}></div>
                <div className="absolute top-40 right-72 w-0.5 h-0.5 bg-blue-50 rounded-full animate-pulse opacity-50" style={{ animationDelay: '3.5s', animationDuration: '1.8s' }}></div>
                
                {/* Shooting stars */}
                <div className="absolute top-6 right-12 w-12 h-0.5 bg-gradient-to-r from-transparent via-cyan-300 to-transparent opacity-80 animate-pulse" style={{ animationDelay: '4s', animationDuration: '6s', transform: 'rotate(-45deg)' }}></div>
                <div className="absolute top-24 left-8 w-8 h-0.5 bg-gradient-to-r from-transparent via-blue-200 to-transparent opacity-70 animate-pulse" style={{ animationDelay: '2s', animationDuration: '5s', transform: 'rotate(-30deg)' }}></div>
                <div className="absolute top-48 right-56 w-10 h-0.5 bg-gradient-to-r from-transparent via-white to-transparent opacity-60 animate-pulse" style={{ animationDelay: '6s', animationDuration: '7s', transform: 'rotate(-60deg)' }}></div>
                
                {/* Constellation connections */}
                <svg className="absolute inset-0 w-full h-full" style={{ opacity: 0.15 }}>
                    <line x1="20%" y1="10%" x2="24%" y2="32%" stroke="cyan" strokeWidth="0.5" opacity="0.3" />
                    <line x1="76%" y1="16%" x2="84%" y2="48%" stroke="blue" strokeWidth="0.5" opacity="0.3" />
                    <line x1="40%" y1="24%" x2="48%" y2="36%" stroke="cyan" strokeWidth="0.3" opacity="0.2" />
                    <line x1="60%" y1="8%" x2="72%" y2="16%" stroke="blue" strokeWidth="0.3" opacity="0.2" />
                </svg>
            </div>
        </div>
    );
}
