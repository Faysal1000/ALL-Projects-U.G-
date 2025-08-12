import { FaArrowDown } from 'react-icons/fa';
import heroImage from '../assets/kahafil-ora-image.png'; 
import dottedArrowIcon from '../assets/vecteezy_hand-drawn-dotted-arrow-line-clip-art_22185812 1.png';

// Configurable Variables
const TEXTURE = 'https://www.transparenttextures.com/patterns/asfalt-dark.png';
const TEXT_COLOR = '#444';
const SKIN_COLOR = "#000000";
const BIBLIOGRAPHY_FONT_FAMILY = "'Plus Jakarta Sans', sans-serif"; 
const FONT_FAMILY = "'Fragment Mono', sans-serif'";
const BIO_TEXT = 'A Visionary Leader With 22+ Years of Experience In Innovation And Growth. As MD Of Goinnovior Limited And Co-Founder of 360D Lifinnovior, And Codeinnovior, He Blends Tech, Business, And Social Impact - Mental Health, And Enabling Free Tech Education.';
const ROLE_TEXT = 'IT CONSULTANT';

const HeroSection = () => {
  return (
    <section
      className="w-full h-[60vh] md:h-[60vh] relative bg-cover bg-no-repeat"
      style={{
        backgroundImage: `url(${TEXTURE})`,
      }}
    >
      <div className="container mx-auto px-0 py-6 relative h-full flex flex-col md:flex-row justify-between items-stretch gap-4">
        {/* Left Column - Hidden on mobile */}
        <div className="flex-1 flex flex-col justify-between hidden md:flex">
          {/* Top Left Name */}
          <div className="font-bold" style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}>
            <h1 className="text-xl md:text-4xl lg:text-6xl xl:text-7xl 2xl:text-8xl uppercase">Kahafil</h1>
            <h1 className="text-lg md:text-xl lg:text-2xl xl:text-3xl 2xl:text-4xl">2K25</h1>
          </div>

          {/* Vertical Text and Line (Middle Left) */}
          <div
            className="absolute top-3/5 left-0 -translate-y-1/2"
            style={{ fontFamily: FONT_FAMILY, color: TEXT_COLOR }}
          >
            <div className="flex flex-col items-start">
              <span className="text-sm transform -rotate-90 tracking-widest -ml-10">/PORTFOLIO</span>
              <div className="w-px h-12 bg-current -m-26 ml-1.5"></div>
            </div>
          </div>

          {/* Bottom Left Scroll Indicator */}
          <div
            className="absolute left-0 bottom-0 flex items-center text-sm"
            style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}
          >
            SCROLL DOWN
            <FaArrowDown className="ml-2 animate-bounce" />
          </div>
        </div>

        {/* Middle Column - Image and Role Text */}
        <div className="flex-1 flex flex-col items-center">
          {/* Kahafil Text for Mobile */}
          <div className="md:hidden w-full text-center py-4 font-bold" style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}>
            <h1 className="text-5xl sm:text-6xl md:text-7xl">KAHAFIL</h1>
          </div>

          {/* Image Wrapper */}
          <div className="w-full md:w-auto position-relative md:position-static">
            <img
              src={heroImage}
              alt="Kahafil"
              className="w-full md:w-[70vh] md:mx-auto h-auto grayscale hover:grayscale-0 transition-all duration-500 md:absolute md:bottom-0 md:left-1/2 md:transform md:-translate-x-1/2"
            />

            {/* Role Text with Icon - Mobile Only (center left) */}
            <div
                className="hidden md:flex absolute md:top-[40%] md:left-[30%] md:-translate-x-[30%] md:-translate-y-[40%] items-center text-base lg:text-lg xl:text-xl"
                style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}
                >
                {ROLE_TEXT}
                <img src={dottedArrowIcon} alt="dotted arrow" className="ml-2 w-25 lg:w-30 h-auto" />
            </div>
          </div>

          {/* ORA for Mobile - After Photo */}
          <div
            className="md:hidden self-end pr-4 pt-4 font-bold"
            style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}
          >
            <h1 className="text-5xl sm:text-6xl">ORA</h1>
          </div>
        </div>

        {/* Right Column - Bibliography and ORA (ORA hidden on mobile) */}
<       div className="flex-1 flex flex-col justify-between items-end pr-4 md:pr-0">
            {/* Bibliography */}
            <div className="w-full md:max-w-[30em] text-right font-normal text-xs sm:text-sm md:text-base md:mt-20" style={{ fontFamily: BIBLIOGRAPHY_FONT_FAMILY }}>
                <h1 className="text-base sm:text-lg md:text-xl xl:text-2xl" style={{ color: "rgba(68, 68, 68, 0.50)" }}>BIOGRAPHY</h1>
                <p className="text-xs sm:text-sm md:text-[0.9em] lg:text-base xl:text-lg" style={{ color: TEXT_COLOR }}>{BIO_TEXT}</p>
            </div>

            {/* ORA Text for Desktop */}
            <div
                className="hidden md:block self-end font-bold"
                style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}
            >
                <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl 2xl:text-8xl">ORA</h1>
            </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;