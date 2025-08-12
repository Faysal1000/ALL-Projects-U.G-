import React from "react";
import { FaArrowDown } from "react-icons/fa";
import heroImage from "../assets/kahafil-ora-image.png";
import dottedArrowIcon from "../assets/vecteezy_hand-drawn-dotted-arrow-line-clip-art_22185812 1.png";
import texture from "../assets/texture.svg";

// ---------------------------
// Configurable variables
// ---------------------------
// This section defines customizable elements like background texture, colors, fonts, and text content for easy modification.
//const TEXTURE = "https://www.transparenttextures.com/patterns/asfalt-dark.png";
const TEXT_COLOR = "#444";
const FONT_FAMILY_GENERAL = "'Plus Jakarta Sans', sans-serif";
const KAHAFIL_ORA_FONT_FAMILY = "'Poppins', 'Fragment Mono', sans-serif"; // primary then fallback
const BIO_TEXT =
  "A Visionary Leader With 22+ Years of Experience In Innovation And Growth. As MD Of Goinnovior Limited And Co-Founder of 360D Soul, Lifeinnovior, And Codeinnovior. He Blends Tech, Business, And Social Impact- Delivering ICT And InfoSec Solutions, Promoting Mental Health, And Enabling Free Tech Education.";
const ROLE_TEXT = "IT CONSULTANT";

/**
 * HERO SECTION
 *
 * This component renders a responsive hero section with a left sidebar (name, portfolio label, scroll prompt),
 * a centered middle image with role label, and a right sidebar (biography and 'ORA' text).
 * On mobile, it stacks vertically with specific alignments: KAHAFIL left, ORA right, image full-width at bottom,
 * role left-bottom, scroll at very bottom.
 * On desktop (md+), left and right are absolutely positioned at edges to allow overlapping the middle when screen is narrow,
 * without causing horizontal scroll. Middle is centered with fixed width (non-shrinking), image in the exact middle of the viewport.
 * Responsiveness uses Tailwind breakpoints (xs to 2xl) for sizes. Background texture applied, grayscale image with hover effect.
 * Overlap enabled via absolute positioning of sides relative to the section.
 */
const HeroSection = () => {
  return (
    // This code sets up the top-level section with full width/height, textured background, and hidden horizontal overflow to prevent scroll from overlaps.
    <section
      className="w-full min-h-screen md:h-screen relative bg-cover bg-no-repeat overflow-x-hidden"
      style={{ backgroundImage: `url(${texture})` }}
    >
      {/* This code creates the main container for mobile stacking (flex-col) and desktop centering of middle (flex justify-center). Padding top only, bottom zero for alignment. */}
      <div className="container mx-0 right-0 left-0 pt-0 md:pt-15 pb-0 h-full flex flex-col md:flex md:justify-center items-stretch gap-0 relative">
        {/* ------------------------------------------------------------------
            LEFT COLUMN (absolute on desktop, hidden on mobile)
            - This code positions left absolutely on md+ at left edge, allowing overlap on middle.
            - Contains name/year top, vertical portfolio in middle, scroll at bottom.
            ------------------------------------------------------------------ */}
        <div className="hidden md:block absolute left-0 top-0 bottom-0 flex flex-col px-0 z-10">
          {/* Name and year: stacked, perfectly aligned */}
          <div
            className="flex flex-col items-start"
            style={{
              color: TEXT_COLOR,
              fontFamily: KAHAFIL_ORA_FONT_FAMILY,
            }}
          >
            <h1
              className="mt-20 uppercase font-[700] text-[6rem] sm:text-[2rem] md:text-[5rem] lg:text-[6rem] xl:text-[6.5rem] 2xl:text-[7rem] leading-none"
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

          {/* Vertical /PORTFOLIO text with a line */}
          <div className="absolute left-4 top-3/5 transform -translate-y-1/2 flex items-start pointer-events-none">
            <div className="flex flex-col items-start -ml-8">
              <span
                className="text-xs sm:text-sm tracking-widest transform -rotate-90 -translate-x-6"
                style={{ fontFamily: FONT_FAMILY_GENERAL, color: TEXT_COLOR }}
              >
                /PORTFOLIO
              </span>
              <div
                className="w-px h-24 bg-current ml-5 -mt-40"
                style={{ backgroundColor: "rgba(68,68,68,0.6)" }}
              />
            </div>
          </div>

          {/* This code places the scroll down prompt at the bottom left with no margin. */}
          <div
            className="absolute left-0 bottom-0 flex items-center text-xs mb-4 ml-4"
            style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY_GENERAL }}
          >
            <span className="uppercase tracking-widest">SCROLL DOWN</span>
            <FaArrowDown className="ml-2 animate-bounce" />
          </div>
        </div>

        {/* ------------------------------------------------------------------
            MIDDLE COLUMN (centered, non-shrinking)
            - This code ensures middle is flex-none with max-width, centered via mx-auto and parent justify-center.
            - On mobile: full width, justified end for bottom alignment.
            - Does not shrink due to auto/max-width sizing.
            ------------------------------------------------------------------ */}
        <div
          className="flex-1 flex flex-col items-center justify-end md:justify-center relative md:flex-none"
          style={{ maxWidth: "760px", margin: "0 auto" }}
        >
          {/* This code shows KAHAFIL title on mobile only, justified left. */}
          <div
            className="md:hidden w-full text-left py-2 font-extrabold"
            style={{ color: TEXT_COLOR, fontFamily: KAHAFIL_ORA_FONT_FAMILY }}
          >
            <h1 className="text-[5.5rem] sm:text-[4.2rem] md:text-[6rem] lg:text-[6rem] xl:text-[7rem] 2xl:text-[8rem]">
              KAHAFIL
            </h1>
          </div>

          {/* This code wraps the image and role label, ensuring bottom margin zero. */}
          <figure className="relative w-full flex justify-center items-end mb-0 z-50">
            {/* This code displays the image full-width on mobile, capped on desktop, centered. */}
            <img
              src={heroImage}
              alt="Kahafil portrait"
              className="w-[100vh] max-w-[480px] md:max-w-[680px] lg:max-w-[720px] xl:max-w-[870px] h-auto object-contain grayscale hover:grayscale-0 transition-all duration-500"
              style={{ display: "block" }}
            />

            {/* This code positions the role badge left-bottom on mobile, left-center on desktop with blur on md+. */}
            <figcaption
              className="absolute left-0 top-[45%] md:left-0 md:top-[53%] md:bottom-auto md:-translate-y-1/2 flex items-center space-x-3 px-3 py-1 md:rounded-lg"
              style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY_GENERAL }}
            >
              <span className="text-[0.8rem] sm:text-sm md:text-base lg:text-lg font-medium">
                {ROLE_TEXT}
              </span>
              <img
                src={dottedArrowIcon}
                alt="dotted arrow"
                className="w-14 h-auto md:w-20"
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
            RIGHT COLUMN (absolute on desktop)
            - This code positions right absolutely on md+ at right edge, mid-height, allowing overlap on middle.
            - On mobile: flows in stack, text right-aligned.
            ------------------------------------------------------------------ */}
        <div className="flex-1 flex flex-col justify-between items-end pr-7 md:absolute md:right-0 md:top-1/3 md:-translate-y-1/2 md:pr-0 md:z-10">
          {/* This code displays the biography block, right-aligned, with max-width on desktop. */}
          <div
            className="w-full mr-0 md:max-w-[28rem] text-right font-normal text-xs sm:text-sm md:text-base mt-2 md:mt-20"
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
        {/* This code shows ORA on desktop only, absolutely bottom right aligned. */}
        <div
          className="hidden md:block absolute right-0 bottom-0 font-extrabold mb-6 mr-6 z-20"
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
        {/* This code adds mobile scroll indicator at the very bottom, left-aligned, no bottom margin. */}
        <div
          className="md:hidden flex items-center text-xs mb-0 px-4"
          style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY_GENERAL }}
        >
          <span className="uppercase tracking-widest">SCROLL DOWN</span>
          <FaArrowDown className="ml-2 animate-bounce" />
        </div>
      </div>
      {/* container becomes a responsive flex row on md+ */}
      <div className="container mx-auto pt-0 md:pt-15 pb-0 flex flex-col md:flex-row md:items-start md:justify-between gap-0 relative">
        {/* LEFT (desktop shown) */}
        <div className="hidden md:flex md:flex-col md:items-start md:w-1/4 z-10">
          {/* name/year etc */}
        </div>

        {/* MIDDLE (always centered) */}
        <div className="w-full md:w-1/2 flex flex-col items-center justify-center">
          {/* image + mobile title */}
        </div>

        {/* RIGHT */}
        <div className="hidden md:flex md:flex-col md:items-end md:w-1/4">
          {/* biography, ORA etc */}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
