'use client';
import BlurText from "@/components/BlurText";
import ImageTrail from "@/components/ImageTrail";
import { Button } from "@/components/ui/button";
import { AnimatePresence, motion } from "motion/react";
import { useRouter } from "next/navigation";
import { useState, useRef } from "react";

export default function Home() {
  const router = useRouter();
  const [isExiting, setIsExiting] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleNavigate = () => {
    setIsExiting(true);
    setTimeout(() => {
      router.push('/Dashboard');
    }, 500); // Match the transition duration
  };

  return (
    <AnimatePresence mode="wait">
      {!isExiting && (
        <motion.div
          key="main"
          initial={{ opacity: 0, y: 100 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -100 }}
          transition={{ duration: 0.5 }}
          className="overflow-visible"
        >
          <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black overflow-visible">
            <main className="flex flex-col gap-4 items-center bg-white dark:bg-black w-full overflow-visible">
            <ImageTrail items={['/Balti1.jpg', '/Balti2.jpg', '/Balti3.jpg', '/Balti4.jpg', '/Balti5.jpg',] as string[]} variant={1}/>
              <h1 className="text-7xl font-extrabold">Scrolink</h1>
              <BlurText text="Safe and secure routes in Baltimore" className="text-2xl text-gray-500"/>  
              <Button ref={buttonRef} className="rounded-full px-8 py-6" onClick={handleNavigate}>
                <p className="text-white">Get Started</p>
              </Button>
              
            </main>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
