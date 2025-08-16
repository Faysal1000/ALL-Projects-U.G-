import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

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
    text: "Another testimonial text goes here for testing smooth transitions and animations in the carousel effect. This one is intentionally long so we can test overflow behaviour. If it is longer than the container allows, it will scroll inside the box (or be clamped if you prefer).",
    author: "John Doe",
    designation: "CEO",
    company: "Another Co.",
  },
];

const Testimonial = () => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % testimonials.length);
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  return (
    <section className="w-full h-auto pt-0 flex flex-col">
      <div className="bg-white flex-1 flex py-[5.2%] px-[6%] flex-col justify-start items-start min-h-0">
        <div className="flex flex-col items-center self-stretch">
          <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-lg md:text-lg lg:text-xl font-[400] leading-normal uppercase mb-4">
            Testimonials
          </div>

          {/* FIXED HEIGHT WRAPPER */}
          <div
            className="relative w-full flex justify-center items-center overflow-hidden
                       h-80 md:h-55"
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={index}
                initial={{ x: "100%", opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: "-100%", opacity: 0 }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
                /* absolute positioning prevents content size affecting parent height */
                className="absolute inset-0 flex flex-col justify-center items-center px-6"
              >
                {/* allow internal vertical scroll when text is long */}
                <div
                  className="text-black text-center font-['Poppins'] text-lg md:text-lg lg:text-xl font-[300] leading-relaxed capitalize
                             max-w-full overflow-y-auto max-h-full pb-2"
                  /* style to hide native scrollbar on webkit*/
                  style={{ WebkitOverflowScrolling: "touch" }}
                >
                  "{testimonials[index].text}"
                </div>

                <div className="mt-4 text-center text-[#000] font-Amiri text-xl font-bold leading-normal capitalize">
                  {testimonials[index].author}
                </div>

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
