import { FaArrowRight } from "react-icons/fa";
import CollaborationCarousel from "./CollaborationCarousel";

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

// Data for all the cards in the carousel
const cards = [
  {
    id: 1,
    title: "Faysal Ahmmed",
    img: "https://placehold.co/200",
    desc: "very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  },
  {
    id: 3,
    title: "Cybersecurity",
    img: "https://placehold.co/200",
    desc: "Protect data and systems with practical, audited security controls.",
    link: "#",
  },
  {
    id: 4,
    title: "DevOps & Automation",
    img: "https://placehold.co/200",
    desc: "Streamline delivery pipelines and reduce manual toil.",
    link: "/devops-automation",
  },
  {
    id: 5,
    title: "Digital Transformation Strategy",
    img: "https://placehold.co/200",
    desc: "Helping businesses modernize operations through tailored digital adoption plans—enhancing efficiency, reducing costs, and boosting productivity.",
    link: "#",
  },
];

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
      <CollaborationCarousel cards={cards} />
    </section>
  );
};

export default Collaboration;
