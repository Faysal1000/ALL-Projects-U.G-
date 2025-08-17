const backgroundTextures = {
  diagonalNoise:
    "https://www.transparenttextures.com/patterns/diagonal-noise.png",
  fabricLight:
    "https://www.transparenttextures.com/patterns/45-degree-fabric-light.png",
  elegantGrid:
    "https://www.transparenttextures.com/patterns/small-criss-cross.png",
  subtleWhite:
    "https://www.transparenttextures.com/patterns/subtle-white-feathers.png",
  brushedMetal: "https://www.transparenttextures.com/patterns/brushed-alum.png",
  lightLeather:
    "https://www.transparenttextures.com/patterns/light-leather.png",
  wavePattern: "https://www.transparenttextures.com/patterns/wavecut.png",
  linen: "https://www.transparenttextures.com/patterns/linen.png",
};

const MetricMarvels = () => {
  return (
    <section
      className="lg:mb-10 bg-cover bg-no-repeat overflow-x-hidden flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-[40px] min-h-0"
      style={{ backgroundImage: `url("${backgroundTextures.fabricLight}")` }}
    >
      {/* Section Title */}
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
        metric marvels
        <span className="text-[#9747FF]">.</span>
      </div>

      <div className="flex flex-col lg:flex-row justify-between items-start self-stretch gap-6">
        {/* Left Content */}
        <div className="flex flex-col justify-center items-start flex-shrink-0">
          <div className="flex items-center gap-[8px]">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="50"
              height="50"
              viewBox="0 0 60 60"
              fill="none"
            >
              <path
                d="M54.5955 27.8158L54.0225 33.904C53.0782 43.9355 52.6062 48.9512 49.6497 51.9755C46.6932 55 42.262 55 33.3997 55H26.6002C17.7379 55 13.3068 55 10.3502 51.9755C7.39369 48.9512 6.92162 43.9355 5.97749 33.904L5.40452 27.8158C4.95449 23.0343 4.72949 20.6435 5.54747 19.6552C5.98994 19.1206 6.59164 18.793 7.23497 18.7365C8.42419 18.632 9.91769 20.3322 12.9047 23.7327C14.4494 25.4912 15.2218 26.3705 16.0834 26.5068C16.5608 26.582 17.0473 26.5045 17.4881 26.2827C18.2838 25.8822 18.8142 24.7953 19.8752 22.6213L25.4672 11.1621C27.472 7.05405 28.4745 5 30 5C31.5255 5 32.528 7.05405 34.5327 11.1621L40.1247 22.6213C41.1857 24.7953 41.7162 25.8822 42.5117 26.2827C42.9527 26.5045 43.4392 26.582 43.9165 26.5068C44.7782 26.3705 45.5505 25.4912 47.0952 23.7327C50.0822 20.3322 51.5757 18.632 52.765 18.7365C53.4082 18.793 54.01 19.1206 54.4525 19.6552C55.2705 20.6435 55.0455 23.0343 54.5955 27.8158Z"
                fill="#444444"
                fillOpacity="0.5"
              />
              <path
                d="M20.625 45C20.625 43.9645 21.4645 43.125 22.5 43.125H37.5C38.5355 43.125 39.375 43.9645 39.375 45C39.375 46.0355 38.5355 46.875 37.5 46.875H22.5C21.4645 46.875 20.625 46.0355 20.625 45Z"
                fill="#444444"
              />
            </svg>
            <div className="text-[#444] font-['Fragment_Mono'] text-base uppercase max-w-[120px]">
              Awards & Recognition
            </div>
          </div>
          <div className="text-[#444] font-[Poppins] text-7xl font-semibold leading-normal uppercase">
            50+
          </div>
        </div>

        {/* Right Table (side scroll on mobile) */}
        <div className="w-full md:w-3/5 overflow-x-auto">
          <div className="min-w-[600px] flex flex-col items-start">
            {/* Row 1 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl">
                Goinnovior Limited
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm">
                2016-Present
              </div>
            </div>

            {/* Row 2 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl">
                360D Soul Limited
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm">
                2023-Present
              </div>
            </div>

            {/* Row 3 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl">
                CodeInnovior
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm">
                2020-Present
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default MetricMarvels;
