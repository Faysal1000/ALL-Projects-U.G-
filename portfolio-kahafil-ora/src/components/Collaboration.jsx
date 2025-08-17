import CollaborationCarousel from "./CollaborationCarousel";
import { COLLABORATION_AND_INNOVATION_ARTICLES } from "/src/config/websiteArticleContent";
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

const Collaboration = () => {
  return (
    <section
      className="bg-cover bg-no-repeat overflow-x-hidden flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-[5.2%] min-h-0"
      style={{ backgroundImage: `url("${backgroundTextures.fabricLight}")` }}
    >
      {/* Wrapper: fills section (flex-1), starts at top (justify-start), allows children to shrink (min-h-0) */}
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
        Creating Excellence Through Collaboration and Innovation
        <span className="text-[#9747FF]">.</span> {/* different color '.' */}
      </div>

      {/* Divider */}
      <div className="self-stretch py-[4.2%]">
        <div className="border-t border-[#444] w-[40%]"></div>
        <div className="border-t border-[#444]"></div>
      </div>

      {/* Carousel Section */}
      <CollaborationCarousel cards={COLLABORATION_AND_INNOVATION_ARTICLES} />
    </section>
  );
};

export default Collaboration;
