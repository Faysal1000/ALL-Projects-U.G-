import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import TESTIMONIALS from "/src/config/testimonialConfig.js";

const Testimonial = () => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % TESTIMONIALS.length);
    }, 5000); // 5 second timer (5000ms) if need slow animation then increase value here ;)
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
                  "{TESTIMONIALS[index].text}"
                </div>

                <div className="mt-4 text-center text-[#000] font-Amiri text-xl font-bold leading-normal capitalize">
                  {TESTIMONIALS[index].author}
                </div>

                <div className="text-[#000] font-Amiri text-sm font-normal leading-normal capitalize">
                  {TESTIMONIALS[index].designation},{" "}
                  <span className="text-[#9747FF]">
                    {TESTIMONIALS[index].company}
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
