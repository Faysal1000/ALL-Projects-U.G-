import { useState } from "react";
import BlogsCarousel from "./BlogsCarousel";
import { BLOGS_AND_ARTICLES } from "/src/config/websiteArticleContent";
import { ARTICLES_GENRES } from "/src/config/websiteArticleContent";

const BlogsAndArticles = () => {
  const [activeGenre, setActiveGenre] = useState(ARTICLES_GENRES[0]); // Default to first genre

  // Filtered cards based on activeGenre
  const filteredCards = BLOGS_AND_ARTICLES.filter(
    (card) => card.genre === activeGenre
  );

  return (
    <section className="bg-[#fff] pt-15 md:pt-0 flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-8 min-h-0">
      {/* Section heading */}
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
        Some solutions that I created
        <span className="text-[#9747FF]">.</span>
      </div>

      {/* Header navigations */}
      <div className="flex justify-between items-center self-stretch flex-wrap">
        {ARTICLES_GENRES.map((genre, idx) => (
          <div key={genre} className="flex items-center">
            <button
              onClick={() => setActiveGenre(genre)}
              className={`text-[#444] font-[Fragment_Mono] text-sm font-normal leading-normal uppercase pb-1 
                        border-b-2 transition-all duration-300 cursor-pointer
                        ${activeGenre === genre ? "border-[#444]/75" : "border-transparent hover:border-[#444]/50 text-[#444]/50"}`}
            >
              {genre}
            </button>

            {idx < ARTICLES_GENRES.length - 1 && (
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
