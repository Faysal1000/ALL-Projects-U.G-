import { useState, useRef, useEffect } from "react";
import { FaArrowRight, FaRegCalendarAlt } from "react-icons/fa";

/**
 * Reusable BlogGrid Component
 * @param {Array} cards - Array of card objects (id, img, type, date, title, desc, link)
 * @param {number} initialCount - Number of cards visible by default (default = 3)
 */
const BlogGrid = ({ cards, initialCount = 3 }) => {
  const [showAll, setShowAll] = useState(false);
  const contentRef = useRef(null);
  const [maxHeight, setMaxHeight] = useState("0px");

  // update height for "extra cards"
  useEffect(() => {
    if (contentRef.current) {
      setMaxHeight(showAll ? `${contentRef.current.scrollHeight}px` : "0px");
    }
  }, [showAll, cards]);

  // Small reusable card renderer
  const BlogCard = ({ card }) => (
    <div className="flex flex-col items-start gap-[10px] flex-shrink-0">
      {/* Card Image */}
      <img
        src={card.img}
        alt={card.title}
        className="w-full rounded-[10px] object-cover"
        style={{
          height: window.innerWidth >= 768 ? "250px" : "200px",
        }}
      />

      {/* Meta Info */}
      <div className="flex justify-between items-center w-full">
        <div className="text-[rgba(68,68,68,0.5)] font-poppins text-sm font-medium capitalize">
          {card.type}
        </div>
        <div className="flex items-center gap-1 text-[rgba(68,68,68,0.5)] font-poppins text-sm font-medium capitalize">
          <FaRegCalendarAlt className="text-[20px]" />
          <span>{card.date}</span>
        </div>
      </div>

      {/* Title */}
      <div className="self-stretch text-[#444] font-Poppins text-lg md:text-2xl font-[700] capitalize leading-normal">
        {card.title}
      </div>

      {/* Description */}
      <div className="self-stretch text-[#444] font-Poppins text-[15px] font-light leading-normal capitalize overflow-hidden text-ellipsis line-clamp-2">
        {card.desc}
      </div>

      {/* Learn More */}
      <a
        href={card.link}
        className="flex items-center gap-2 text-[#444] font-Poppins text-[15px] font-medium capitalize cursor-pointer hover:text-[#9747FF]"
      >
        <span>Learn more</span>
        <FaArrowRight size={14} />
      </a>
    </div>
  );

  return (
    <div className="w-full">
      {/* Always show first N blogs */}
      <div className="grid gap-6 md:gap-8 grid-cols-1 md:grid-cols-3">
        {cards.slice(0, initialCount).map((card) => (
          <BlogCard key={card.id} card={card} />
        ))}
      </div>

      {/* Expandable section for rest */}
      {cards.length > initialCount && (
        <div
          className="overflow-hidden transition-all duration-700 ease-in-out"
          style={{ maxHeight }}
          ref={contentRef}
        >
          <div className="grid gap-6 md:gap-8 grid-cols-1 md:grid-cols-3 mt-6">
            {cards.slice(initialCount).map((card) => (
              <BlogCard key={card.id} card={card} />
            ))}
          </div>
        </div>
      )}

      {/* Load More / See Less Button */}
      {cards.length > initialCount && (
        <div className="flex justify-center mt-6">
          <button
            onClick={() => setShowAll(!showAll)}
            className="px-6 py-2 rounded-sm bg-[#444444] text-white font-poppins text-sm md:text-base font-medium hover:bg-[#000000] transition ease-in-out duration-700"
          >
            {showAll ? "See Less" : "Load More"}
          </button>
        </div>
      )}
    </div>
  );
};

export default BlogGrid;
