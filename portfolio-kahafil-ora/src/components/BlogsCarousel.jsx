import React, { useRef, useState, useEffect } from "react";
import { FaArrowRight, FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { FaRegCalendarAlt } from "react-icons/fa";

const BlogsCarousel = ({ cards }) => {
  // Reference to the carousel container for measuring visible area
  const containerRef = useRef(null);
  // Reference to the first card to measure its width (including gap)
  const firstCardRef = useRef(null);

  // State for full card width (card width + gap)
  const [cardFullWidth, setCardFullWidth] = useState(0);
  // State for number of visible cards in the viewport
  const [visibleCount, setVisibleCount] = useState(1);
  // State for current scroll index (starting card)
  const [index, setIndex] = useState(0);
  // State for drag offset during swipe/drag for live preview
  const [offset, setOffset] = useState(0);
  // State to track if dragging is in progress (to disable transitions)
  const [isDragging, setIsDragging] = useState(false);

  // Gap between cards in pixels
  const GAP_PX = 15;
  // Threshold fraction for swipe to trigger move (e.g., 1/100 of card width)
  // The lower the threshold the smoother the swipe
  const SWIPE_THRESHOLD_FRACTION = 1 / 100;

  // Refs for touch/pointer tracking
  const touchStartX = useRef(0);
  const touchDeltaX = useRef(0);
  const pointerDown = useRef(false);

  // Calculate arrow move steps (page-like on desktop, 1 on mobile)
  const arrowMoveBy = visibleCount > 1 ? visibleCount - 1 : 1;
  // Swipe always moves by 1 card for finer control, especially on mobile
  const swipeMoveBy = 1;

  // Effect to measure card width and visible count on mount/resize
  useEffect(() => {
    const measure = () => {
      const container = containerRef.current;
      const firstCard = firstCardRef.current;
      if (!container || !firstCard) return;

      // Get card width + gap
      const totalW = Math.round(
        firstCard.getBoundingClientRect().width + GAP_PX
      );
      setCardFullWidth(totalW);

      // Calculate visible cards based on container width
      const containerW = container.getBoundingClientRect().width;
      const count = Math.max(1, Math.floor(containerW / totalW));
      setVisibleCount(count);

      // Clamp index to valid range
      const maxIdx = Math.max(0, cards.length - count);
      setIndex((cur) => Math.min(cur, maxIdx));
    };

    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Max index to prevent scrolling past end
  const maxIndex = Math.max(0, cards.length - visibleCount);

  // Function to navigate left (previous cards)
  const goLeft = (steps = 1) => {
    setIndex((cur) => Math.max(0, cur - steps));
    setOffset(0); // Reset offset after move
  };

  // Function to navigate right (next cards)
  const goRight = (steps = 1) => {
    setIndex((cur) => Math.min(maxIndex, cur + steps));
    setOffset(0); // Reset offset after move
  };

  // Calculate base translateX based on index + drag offset
  const translateX = -index * cardFullWidth + offset;

  // Style for the track with conditional transition for smooth drag
  const trackStyle = {
    transform: `translateX(${translateX}px)`,
    transition: isDragging ? "none" : "transform 400ms ease",
  };

  // --- Touch handlers for mobile swipe ---
  const handleTouchStart = (e) => {
    if (!e.touches || e.touches.length === 0) return;
    touchStartX.current = e.touches[0].clientX;
    touchDeltaX.current = 0;
    setIsDragging(true);
  };

  const handleTouchMove = (e) => {
    if (!e.touches || e.touches.length === 0 || !isDragging) return;
    touchDeltaX.current = e.touches[0].clientX - touchStartX.current;
    setOffset(touchDeltaX.current); // Live update offset for smooth drag
  };

  const handleTouchEnd = () => {
    if (!isDragging) return;
    setIsDragging(false);
    const dx = touchDeltaX.current;
    const threshold = cardFullWidth * SWIPE_THRESHOLD_FRACTION;
    if (dx > threshold) {
      // Swipe right -> go to previous
      goLeft(swipeMoveBy);
    } else if (dx < -threshold) {
      // Swipe left -> go to next
      goRight(swipeMoveBy);
    } else {
      // Snap back if not enough swipe
      setOffset(0);
    }
    touchDeltaX.current = 0;
  };

  // --- Pointer handlers for desktop mouse drag ---
  const handlePointerDown = (e) => {
    pointerDown.current = true;
    touchStartX.current = e.clientX;
    touchDeltaX.current = 0;
    setIsDragging(true);
    if (e.target && e.target.setPointerCapture) {
      try {
        e.target.setPointerCapture(e.pointerId);
      } catch {}
    }
  };

  const handlePointerMove = (e) => {
    if (!pointerDown.current || !isDragging) return;
    touchDeltaX.current = e.clientX - touchStartX.current;
    setOffset(touchDeltaX.current); // Live update offset for smooth drag
  };

  const handlePointerUp = (e) => {
    if (!pointerDown.current || !isDragging) return;
    pointerDown.current = false;
    setIsDragging(false);
    const dx = touchDeltaX.current;
    const threshold = cardFullWidth * SWIPE_THRESHOLD_FRACTION;
    if (dx > threshold) {
      goLeft(swipeMoveBy);
    } else if (dx < -threshold) {
      goRight(swipeMoveBy);
    } else {
      setOffset(0);
    }
    touchDeltaX.current = 0;
    if (e.target && e.target.releasePointerCapture) {
      try {
        e.target.releasePointerCapture(e.pointerId);
      } catch {}
    }
  };

  const handlePointerLeave = () => {
    if (!pointerDown.current || !isDragging) return;
    pointerDown.current = false;
    setIsDragging(false);
    const dx = touchDeltaX.current;
    const threshold = cardFullWidth * SWIPE_THRESHOLD_FRACTION;
    if (dx > threshold) {
      goLeft(swipeMoveBy);
    } else if (dx < -threshold) {
      goRight(swipeMoveBy);
    } else {
      setOffset(0);
    }
    touchDeltaX.current = 0;
  };

  return (
    <div className="w-full pt-15">
      {/* Carousel Wrapper */}
      <div className="relative w-full">
        {/* Left Navigation Arrow (hidden on mobile) */}
        <button
          onClick={() => goLeft(arrowMoveBy)}
          className="hidden md:flex items-center justify-center absolute left-0 top-1/3 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white shadow"
        >
          <FaChevronLeft />
        </button>

        {/* Right Navigation Arrow (hidden on mobile) */}
        <button
          onClick={() => goRight(arrowMoveBy)}
          className="hidden md:flex items-center justify-center absolute right-0 top-1/3 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white shadow"
        >
          <FaChevronRight />
        </button>

        {/* Viewport (shows visible cards) */}
        <div
          ref={containerRef}
          className="overflow-hidden w-full"
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerLeave}
        >
          {/* Track (scrollable card list) */}
          <div
            className="flex w-full items-start gap-[15px]"
            style={trackStyle}
          >
            {cards.map((card, i) => (
              <div
                key={card.id}
                ref={i === 0 ? firstCardRef : null}
                className="flex flex-col items-start gap-[10px] w-[300px] md:w-[500px] flex-shrink-0"
              >
                {/* Card Image */}
                <img
                  src={card.img}
                  alt={card.title}
                  className="h-[220px] md:h-[300px] w-full rounded-[10px] object-cover"
                />

                <div className="flex justify-between items-center w-full">
                  {/* Left text */}
                  <div className="text-[rgba(68,68,68,0.5)] font-poppins text-sm font-medium leading-normal capitalize">
                    {card.type}
                  </div>

                  {/* Right text with icon */}
                  <div className="flex items-center gap-1 text-[rgba(68,68,68,0.5)] font-poppins text-sm font-medium leading-normal capitalize">
                    <FaRegCalendarAlt className="text-[20px]" />
                    <span>{card.date}</span>
                  </div>
                </div>

                {/* Card Title */}
                <div className="self-stretch text-[#444] font-Poppins text-lg md:text-2xl font-[700] capitalize leading-normal">
                  {card.title}
                </div>

                {/* Card Description */}
                <div className="self-stretch text-[#444] font-Poppins text-[15px] font-light leading-normal capitalize overflow-hidden text-ellipsis line-clamp-2">
                  {card.desc}
                </div>

                {/* Learn More Link */}
                <a
                  href={card.link}
                  className="flex items-center gap-2 text-[#444] font-Poppins text-[15px] font-medium leading-normal capitalize cursor-pointer hover:text-[#9747FF]"
                >
                  <span>Learn more</span>
                  <FaArrowRight size={14} />
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BlogsCarousel;
