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

const educationData = [
  {
    year: "2019",
    title: "Bachelor of Science",
    institution: "Daffodil International University",
  },
  {
    year: "2014",
    title: "Higher School Certificate",
    institution: "Daffodil International School",
  },
  {
    year: "2012",
    title: "Secondary School Certificate",
    institution: "Daffodil International School",
  },
  {
    year: "2001",
    title: "Microsoft Certified Architect",
    institution: "Microsoft",
  },
  {
    year: "2009",
    title: "Certification in Graphic Design",
    institution: "ABC Design Academy",
  },
  {
    year: "2008",
    title: "Professional IT Essentials",
    institution: "Tech Learning Hub",
  },
  {
    year: "2007",
    title: "Leadership & Management Program",
    institution: "Global Institute",
  },
  {
    year: "2006",
    title: "Entrepreneurship Development",
    institution: "Startup Academy",
  },
  {
    year: "2005",
    title: "Advanced English Communication",
    institution: "Language Pro Center",
  },
];

const Education = () => {
  return (
    <section
      className="lg:mb-10 bg-cover bg-no-repeat overflow-x-hidden flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-[5.2%] min-h-0"
      style={{ backgroundImage: `url("${backgroundTextures.fabricLight}")` }}
    >
      <div className="text-[#444] text-center font-[Amiri] text-3xl sm:text-2xl md:text-4xl lg:text-5xl not-italic font-normal leading-normal capitalize self-stretch mb-5 md:mb-10">
        Education & Certifications
      </div>

      {/* 
        Grid:
        - Mobile: 1 column (stacked)
        - LG+: 3 columns, 3 rows, and COLUMN flow -> first 3 items fill the first column, next 3 the second, etc.
      */}
      <div
        className="
          grid 
          grid-cols-1 
          sm:grid-cols-2 
          lg:grid-cols-3 
          lg:grid-rows-3 
          lg:grid-flow-col 
          gap-[40px] md:gap-[70px] 
          self-stretch
        "
      >
        {educationData.map((edu, idx) => (
          <div key={idx} className="flex flex-col justify-center items-start">
            <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-sm not-italic font-normal leading-normal uppercase">
              {edu.year}
            </div>
            <div className="text-black font-[Amiri] text-xl not-italic font-normal leading-[200%] tracking-[0.4px] capitalize">
              {edu.title}
            </div>
            <div className="text-[#444] font-[Poppins] text-base not-italic font-light leading-[200%] tracking-[0.3px] capitalize">
              {edu.institution}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Education;
