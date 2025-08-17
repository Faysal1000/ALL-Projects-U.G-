import { FaArrowRight, FaRegCalendarAlt } from "react-icons/fa";
import { useState, useEffect } from "react";
import Footer from "./components/Footer";
import BlogGrid from "./components/BlogGrid.jsx";
import {
  THOUGHT_SECTION_MAIN_NEWS,
  BLOGS_AND_ARTICLES,
  PUBLIC_SPEAKING_AND_TRAINING_ARTICLES,
  RECENT_EVENT_AND_ACTIVITIES,
  SOCIAL_ACTIVITIES,
} from "./config/websiteArticleContent.js";

const genres = [
  "Blogs & Articles",
  "Public Speaking & Training",
  "Recent Event Activities",
  "Social Activities",
];

const Thoughts = () => {
  const [showAll, setShowAll] = useState(false);
  const [initialCount, setInitialCount] = useState(3);

  // Detect screen size and set initialCount
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setInitialCount(1); // mobile: show 1 blog
      } else {
        setInitialCount(3); // desktop: show 3 blogs
      }
    };

    handleResize(); // run once on mount
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <section className="w-full min-h-screen flex flex-col">
      <div
        className="flex flex-col mt-15 justify-end p-10 md:p-[80px] gap-[10px] self-stretch rounded-[20px] 
               bg-no-repeat min-h-[60vh] md:min-h-screen
               bg-contain md:bg-cover bg-top"
        style={{
          backgroundImage: `linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.8) 100%), url(${THOUGHT_SECTION_MAIN_NEWS.image})`,
          backgroundColor: "#d3d3d3",
        }}
      >
        {/* Top row */}
        <div className="flex justify-between items-center w-full text-white/50 font-poppins text-base md:text-xl font-medium leading-normal capitalize">
          <div>{THOUGHT_SECTION_MAIN_NEWS.category}</div>
          <div className="flex items-center space-x-2">
            <FaRegCalendarAlt />
            <span>{THOUGHT_SECTION_MAIN_NEWS.date}</span>
          </div>
        </div>

        {/* Headline */}
        <div className="text-white font-poppins text-2xl lg:text-3xl 2xl:text-4xl font-bold leading-normal capitalize">
          {THOUGHT_SECTION_MAIN_NEWS.title}
        </div>

        {/* Description */}
        <div
          className="self-stretch text-white font-poppins font-light leading-normal capitalize 
                 text-base md:text-lg lg:text-xl"
        >
          {THOUGHT_SECTION_MAIN_NEWS.description}
        </div>

        {/* Link */}
        <a
          href={THOUGHT_SECTION_MAIN_NEWS.link}
          className="flex items-center gap-2 text-[#FFFFFF] font-poppins text-[15px] font-medium leading-normal capitalize cursor-pointer hover:text-[#9747FF]"
        >
          <span>Read on Facebook</span>
          <FaArrowRight size={14} />
        </a>
      </div>

      {/*blogs and article section */}
      <div className="bg-[#fff] pt-15 flex-1 flex py-[5.2%] md:py-5 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Blogs & Articles
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid cards={BLOGS_AND_ARTICLES} initialCount={initialCount} />
      </div>

      {/*Public Speaking & Training.*/}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] md:py-5 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Public Speaking & Training
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid
          cards={PUBLIC_SPEAKING_AND_TRAINING_ARTICLES}
          initialCount={initialCount}
        />
      </div>

      {/*Recent Event Activities.*/}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] md:py-10 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Recent Event Activities
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid
          cards={RECENT_EVENT_AND_ACTIVITIES}
          initialCount={initialCount}
        />
      </div>

      {/*Social Activities.*/}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] md:py-10 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Social Activities
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid cards={SOCIAL_ACTIVITIES} initialCount={initialCount} />
      </div>
      <Footer />
    </section>
  );
};

export default Thoughts;
