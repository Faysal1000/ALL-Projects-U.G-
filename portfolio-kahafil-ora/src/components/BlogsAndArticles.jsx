import { useState } from "react";
import BlogsCarousel from "./BlogsCarousel";

// Data for all the cards in the carousel
const myCards = [
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

const genres = [
  "Blogs & Articles",
  "Public Speaking & Training",
  "Recent Event Activities",
  "Social Activities",
];

const BlogsAndArticles = () => {
  const [activeGenre, setActiveGenre] = useState("Blogs & Articles");

  // Filtered cards based on activeGenre
  const filteredCards = myCards.filter((card) => card.genre === activeGenre);

  return (
    <section className="bg-[#fff] pt-15 md:pt-0 flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-8 min-h-0">
      {/* Section heading */}
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
        Some solutions that I created
        <span className="text-[#9747FF]">.</span>
      </div>

      {/* Header navigations */}
      <div className="flex justify-between items-center self-stretch flex-wrap">
        {genres.map((genre, idx) => (
          <div key={genre} className="flex items-center">
            <button
              onClick={() => setActiveGenre(genre)}
              className={`text-[#444] font-[Fragment_Mono] text-sm font-normal leading-normal uppercase pb-1 
                        border-b-2 transition-all duration-300 cursor-pointer
                        ${activeGenre === genre ? "border-[#444]/75" : "border-transparent hover:border-[#444]/50 text-[#444]/50"}`}
            >
              {genre}
            </button>

            {idx < genres.length - 1 && (
              <span className="mx-2 text-[#444] font-[Fragment_Mono]">/</span>
            )}
          </div>
        ))}
      </div>

      {/* Blogs carousel with filtered data */}
      <BlogsCarousel
        cards={filteredCards}
        desktopImageWidth="500px"
        desktopImageHeight="300px"
        mobileImageWidth="300px"
        mobileImageHeight="220px"
      />
    </section>
  );
};

export default BlogsAndArticles;
