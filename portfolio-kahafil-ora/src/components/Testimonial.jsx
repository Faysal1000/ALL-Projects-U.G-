import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Array of testimonial objects
const testimonials = [
  {
    text: "Kahafil Ora is an experienced IT Consultant known for delivering smart, tech-driven solutions to businesses. With a strong grasp of IT infrastructure and digital strategy, he helps organizations improve efficiency and achieve their goals through innovative technology.",
    author: "kahafil ora",
    designation: "CTO",
    company: "Company name",
  },
  {
    text: "Faysal Ahmmed is a dedicated researcher and software engineer with deep expertise in AI, deep learning, and backend development. He combines academic excellence with practical problem-solving skills.",
    author: "Faysal Ahmmed",
    designation: "Researcher",
    company: "AIUB",
  },
  {
    text: "Another testimonial text goes here for testing smooth transitions and animations in the carousel effect.",
    author: "John Doe",
    designation: "CEO",
    company: "Another Co.",
  },
];

const Testimonial = () => {
  // State to track the current testimonial index
  const [index, setIndex] = useState(0);

  // useEffect to automatically change testimonial every 5 seconds
  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % testimonials.length); // loop back to first testimonial
    }, 5000); // 5000ms = 5s
    return () => clearInterval(timer); // cleanup interval on unmount
  }, []);

  return (
    <section className="w-full min-h-screen pt-0 flex flex-col">
      {/* Main container */}
      <div className="bg-[#fff] pt-15 flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start min-h-0">
        <div className="flex flex-col items-center self-stretch">
          {/* Section title */}
          <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-lg md:text-lg lg:text-xl font-[400] leading-normal uppercase">
            Testimonials
          </div>

          {/* Motion container for sliding testimonials */}
          <div className="relative w-full flex justify-center items-center overflow-hidden">
            <AnimatePresence mode="wait">
              {/* Motion.div handles the animation of entering/exiting testimonials */}
              <motion.div
                key={index} // re-renders animation when index changes
                initial={{ x: "100%", opacity: 0 }} // start off-screen right
                animate={{ x: 0, opacity: 1 }} // animate to center
                exit={{ x: "-100%", opacity: 0 }} // exit off-screen left
                transition={{ duration: 0.8, ease: "easeInOut" }} // smooth animation
                className="w-full flex flex-col items-center" // center content
              >
                {/* Testimonial text */}
                <div className="text-black text-center font-['Poppins'] pt-10 text-lg md:text-lg lg:text-xl font-[300] leading-normal capitalize">
                  "{testimonials[index].text}"
                </div>

                {/* Author name */}
                <div className="text-center text-[#000] font-Amiri text-xl font-bold leading-normal capitalize">
                  {testimonials[index].author}
                </div>

                {/* Designation and company */}
                <div className="text-[#000] font-Amiri text-sm font-normal leading-normal capitalize">
                  {testimonials[index].designation},{" "}
                  <span className="text-[#9747FF]">
                    {testimonials[index].company}
                  </span>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonial;
