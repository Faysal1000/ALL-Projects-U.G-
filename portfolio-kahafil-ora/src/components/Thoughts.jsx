import { FaArrowRight, FaRegCalendarAlt } from "react-icons/fa";
import BlogGrid from "./BlogGrid";
import { useState, useEffect } from "react";
import Footer from "./Footer";

// main news section
const mainNews = {
  title: "আইটি প্রফেশনাল মিটআপ",
  description:
    "দেশের আইটি খাতের ১৬০ জন আইটি প্রফেশনালকে বিশ্বমানের দক্ষতা উন্নয়ন প্রশিক্ষণ দেওয়া হয়েছে। সম্প্রতি প্রশিক্ষণ শেষে ঢাকা বিশ্ববিদ্যালয়ের আইবিএ বিভাগে ACMP 4.0 মিটআপ ২০২৪ অনুষ্ঠিত হয়।",
  date: "May 25, 2025",
  category: "Education, Social Impact",
  link: "https://www.facebook.com/events/123456789",
  image: "src/assets/thoughts_image.jpg",
};

//blogs and articles section
const blogsAndArticles = [
  {
    id: 1,
    type: "Cyber Security",
    genre: "Blogs & Articles",
    date: "15 Aug 2025",
    title:
      "Faysal Ahmmed, A software engineer with excellent skills and academic scores",
    img: "https://placehold.co/200",
    desc: "Very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    type: "Cloud Migration",
    genre: "Public Speaking & Training",
    date: "15 Aug 2025",
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  },
  {
    id: 3,
    type: "Cybersecurity",
    genre: "Recent Event Activities",
    date: "15 Aug 2025",
    title: "Cybersecurity",
    img: "https://placehold.co/200",
    desc: "Protect data and systems with practical, audited security controls.",
    link: "#",
  },
  {
    id: 4,
    type: "DevOps & Automation",
    genre: "Social Activities",
    date: "15 Aug 2025",
    title: "DevOps & Automation",
    img: "https://placehold.co/200",
    desc: "Streamline delivery pipelines and reduce manual toil.",
    link: "/devops-automation",
  },
  {
    id: 5,
    type: "Digital Transformation",
    genre: "Social Activities",
    date: "15 Aug 2025",
    title: "Digital Transformation Strategy",
    img: "https://placehold.co/200",
    desc: "Helping businesses modernize operations through tailored digital adoption plans—enhancing efficiency, reducing costs, and boosting productivity.",
    link: "#",
  },
];
// similarly i have to make object for other sections

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
          backgroundImage: `linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.8) 100%), url(${mainNews.image})`,
          backgroundColor: "#d3d3d3",
        }}
      >
        {/* Top row */}
        <div className="flex justify-between items-center w-full text-white/50 font-poppins text-base md:text-xl font-medium leading-normal capitalize">
          <div>{mainNews.category}</div>
          <div className="flex items-center space-x-2">
            <FaRegCalendarAlt />
            <span>{mainNews.date}</span>
          </div>
        </div>

        {/* Headline */}
        <div className="text-white font-poppins text-2xl lg:text-3xl 2xl:text-4xl font-bold leading-normal capitalize">
          {mainNews.title}
        </div>

        {/* Description */}
        <div
          className="self-stretch text-white font-poppins font-light leading-normal capitalize 
                 text-base md:text-lg lg:text-xl"
        >
          {mainNews.description}
        </div>

        {/* Link */}
        <a
          href={mainNews.link}
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
        <BlogGrid cards={blogsAndArticles} initialCount={initialCount} />
      </div>

      {/*Public Speaking & Training.*/}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] md:py-5 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Public Speaking & Training
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid cards={blogsAndArticles} initialCount={initialCount} />
      </div>

      {/*Recent Event Activities.*/}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] md:py-10 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Recent Event Activities
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid cards={blogsAndArticles} initialCount={initialCount} />
      </div>

      {/*Social Activities.*/}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] md:py-10 px-[12.5%] flex-col justify-start items-start gap-1">
        {/* Section heading */}
        <div className="text-[#444] pb-3 font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
          Social Activities
          <span className="text-[#9747FF]">.</span>
        </div>
        <BlogGrid cards={blogsAndArticles} initialCount={initialCount} />
      </div>
      <Footer />
    </section>
  );
};

export default Thoughts;
