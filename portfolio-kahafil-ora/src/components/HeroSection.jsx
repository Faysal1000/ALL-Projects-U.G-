import React from "react";
import { FaArrowDown } from "react-icons/fa";
import heroImage from "../assets/kahafil-ora-image.png";
import dottedArrowIcon from "../assets/vecteezy_hand-drawn-dotted-arrow-line-clip-art_22185812 1.png";
import texture from "../assets/texture.svg";

// ---------------------------
// Configurable variables
// ---------------------------
// This section defines customizable elements like background texture, colors, fonts, and text content for easy modification.
const TEXT_COLOR = "#444";
const FONT_FAMILY_GENERAL = "'Plus Jakarta Sans', sans-serif";
const KAHAFIL_ORA_FONT_FAMILY = "'Poppins', 'Fragment Mono', sans-serif"; // primary then fallback
const BIO_TEXT =
  "A Visionary Leader With 22+ Years of Experience In Innovation And Growth. As MD Of Goinnovior Limited And Co-Founder of 360D Soul, Lifeinnovior, And Codeinnovior. He Blends Tech, Business, And Social Impact- Delivering ICT And InfoSec Solutions, Promoting Mental Health, And Enabling Free Tech Education.";
const ROLE_TEXT = "IT CONSULTANT";

/**
 * HERO SECTION
 *
 * This component renders a responsive hero section divided into three parts: left sidebar (name, portfolio label, scroll prompt),
 * centered middle (image with role label), and right sidebar (biography and 'ORA' text).
 * On mobile, it stacks vertically with KAHAFIL left, ORA right, image at bottom, role left-bottom, and scroll at bottom.
 * On desktop (md+), it uses a flex-row with left and right taking 1/4 width each, middle 1/2 width, centered and non-shrinking.
 * Responsiveness uses Tailwind breakpoints (xs to 2xl). Background texture applied, grayscale image with hover effect.
 */
const HeroSection = () => {
  return (
    // This code sets up the top-level section with full width/height, textured background, and hidden horizontal overflow.
    <section
      className="w-full h-screen relative bg-cover bg-no-repeat overflow-x-hidden"
      style={{ backgroundImage: `url(${texture})` }}
    >
      {/* This code creates a responsive container that stacks on mobile (flex-col) and becomes a flex row on md+ with justified content. */}
      <div className="container mx-auto pt-0 md:pt-15 pb-0 flex flex-col md:flex-row md:items-start md:justify-between gap-0 relative h-full">
        {/* ------------------------------------------------------------------
            LEFT COLUMN (desktop shown)
            - This code defines the left part, visible on md+, taking 1/4 width, containing name/year, portfolio label, and scroll prompt.
            ------------------------------------------------------------------ */}
        <div className="hidden md:flex md:flex-col md:items-start md:w-1/4 z-10">
          {/* This code displays the name and year at the top left with custom styling. */}
          <div
            className="flex flex-col items-start mt-20"
            style={{
              color: TEXT_COLOR,
              fontFamily: KAHAFIL_ORA_FONT_FAMILY,
            }}
          >
            <h1
              className="uppercase font-[700] text-[6rem] sm:text-[2rem] md:text-[5rem] lg:text-[6rem] xl:text-[6.5rem] 2xl:text-[7rem] leading-none"
              style={{
                WebkitTextStrokeWidth: "5px",
                WebkitTextStrokeColor: "#444",
              }}
            >
              Kahafil
            </h1>
            <h1 className="font-bold text-xs sm:text-sm md:text-base lg:text-lg xl:text-xl leading-none">
              2K25
            </h1>
          </div>

          {/* This code adds the vertical /PORTFOLIO text with a line, positioned mid-height. */}
          <div className="absolute left-4 top-3/5 transform -translate-y-1/2 flex items-start pointer-events-none">
            <div className="flex flex-col items-start -ml-8">
              <span
                className="text-xs sm:text-sm tracking-widest transform -rotate-90 -translate-x-6"
                style={{ fontFamily: FONT_FAMILY_GENERAL, color: TEXT_COLOR }}
              >
                /PORTFOLIO
              </span>
              <div
                className="w-px h-12 bg-current ml-5.5 -mt-28"
                style={{ backgroundColor: "rgba(68,68,68,0.6)" }}
              />
            </div>
          </div>

          {/* This code places the scroll down prompt at the bottom left. */}
          <div
            className="absolute left-0 bottom-0 flex items-center text-xs sm:text-sm md:text-sm mb-4 ml-4"
            style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY_GENERAL }}
          >
            <span className="uppercase tracking-widest">SCROLL DOWN</span>
            <FaArrowDown className="ml-2 animate-bounce" />
          </div>
        </div>

        {/* ------------------------------------------------------------------
            MIDDLE COLUMN (always centered)
            - This code defines the middle part, always centered, taking 1/2 width on md+, full width on mobile, with image at bottom on mobile.
            ------------------------------------------------------------------ */}
        <div className="w-full pt-10 md:pt-0 md:w-1/2 flex flex-col justify-end items-center relative md:flex-none h-full">
          {/* This code shows KAHAFIL title on mobile only, justified left. */}
          <div
            className="md:hidden w-full text-left py-2 font-[700]"
            style={{
              color: TEXT_COLOR,
              fontFamily: KAHAFIL_ORA_FONT_FAMILY,
              WebkitTextStrokeWidth: "5px",
              WebkitTextStrokeColor: "#444",
            }}
          >
            <h1 className="text-[4.5rem] sm:text-[4.2rem] md:text-[6rem] lg:text-[6rem] xl:text-[7rem] 2xl:text-[8rem]">
              KAHAFIL
            </h1>
          </div>

          {/* This code wraps the image and role label, ensuring bottom margin zero. */}
          <figure className="relative w-full flex justify-center items-end mb-0 z-10">
            {/* This code displays the image full-width on mobile, capped on desktop, centered. */}
            <img
              src={heroImage}
              alt="Kahafil portrait"
              className="z-50 w-[100vh] max-w-[380px] md:max-w-[680px] lg:max-w-[720px] xl:max-w-[870px] h-auto object-contain hover:grayscale-0 transition-all duration-500 md:grayscale
"
              style={{ display: "block" }}
            />

            {/* This code positions the role badge left-bottom on mobile, left-center on desktop. */}
            <figcaption
              className="absolute left-0 top-[45%] md:-left-10 md:top-[54%] md:bottom-auto md:-translate-y-1/2 flex items-center space-x-0 px-3 py-1 md:rounded-lg"
              style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY_GENERAL }}
            >
              <span className="text-[0.6rem] sm:text-sm md:text-base lg:text-lg font-medium">
                {ROLE_TEXT}
              </span>
              <img
                src={dottedArrowIcon}
                alt="dotted arrow"
                className="w-12 h-auto md:w-20"
              />
            </figcaption>
          </figure>

          {/* This code shows ORA on mobile only, justified right. */}
          <div className="w-full flex justify-end mr-15 mt-4 md:hidden">
            <h2
              className="font-extrabold"
              style={{
                color: TEXT_COLOR,
                fontFamily: KAHAFIL_ORA_FONT_FAMILY,
                fontSize: "4.5rem",
                lineHeight: 1,
                WebkitTextStrokeWidth: "5px",
                WebkitTextStrokeColor: "#444",
              }}
            >
              ORA
            </h2>
          </div>
        </div>

        {/* ------------------------------------------------------------------
            RIGHT COLUMN
            - This code defines the right part, visible on md+, taking 1/4 width, containing biography and ORA.
            ------------------------------------------------------------------ */}
        <div className="hidden md:flex md:flex-col md:items-end md:w-1/4">
          {/* This code displays the biography block, right-aligned, with max-width. */}
          <div
            className="w-[140%] md:max-w-[28rem] text-right font-normal text-xs sm:text-sm md:text-base mt-2 md:mt-20"
            style={{ fontFamily: FONT_FAMILY_GENERAL }}
          >
            <h3
              className="text-xs sm:text-sm md:text-base mb-2"
              style={{ color: "rgba(68,68,68,0.6)" }}
            >
              BIOGRAPHY
            </h3>
            <p
              className="text-black text-right font-plusjakarta text-[0.85rem] font-normal leading-[1.6] tracking-[0.2px] capitalize"
              style={{ color: TEXT_COLOR }}
            >
              {BIO_TEXT}
            </p>
          </div>
        </div>

        {/* ============================
            ORA — ABSOLUTE BOTTOM-RIGHT DESKTOP
            Positioned outside right column so it's pinned
        ============================ */}
        <div
          className="hidden md:block absolute bottom-0 right-0 font-extrabold mb-6 mr-6 z-10"
          style={{
            color: TEXT_COLOR,
            fontFamily: KAHAFIL_ORA_FONT_FAMILY,
            WebkitTextStrokeWidth: "5px",
            WebkitTextStrokeColor: "#444",
          }}
        >
          <h1 className="uppercase font-[700] text-[6rem] sm:text-[2rem] md:text-[5rem] lg:text-[6rem] xl:text-[6.5rem] 2xl:text-[7.0rem] leading-none">
            ORA
          </h1>
        </div>

        {/* Mobile Biography - Appears after ORA and is right-aligned */}
        <div className="md:hidden flex flex-col items-end w-full px-4 mt-4 mr-5">
          <div
            className="w-full text-right font-normal text-xs sm:text-sm"
            style={{ fontFamily: FONT_FAMILY_GENERAL }}
          >
            <h3
              className="text-xs sm:text-sm mb-2"
              style={{ color: "rgba(68,68,68,0.6)" }}
            >
              BIOGRAPHY
            </h3>
            <p
              className="text-black font-plusjakarta text-[0.85rem] font-normal leading-[1.6] tracking-[0.2px] capitalize"
              style={{ color: TEXT_COLOR }}
            >
              {BIO_TEXT}
            </p>
          </div>
        </div>

        {/* This code adds mobile scroll indicator at the very bottom, left-aligned, no bottom margin. */}
        <div
          className="md:hidden flex items-center text-xs mb-0 px-4"
          style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY_GENERAL }}
        >
          <span className="uppercase tracking-widest">SCROLL DOWN</span>
          <FaArrowDown className="ml-2 animate-bounce" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
