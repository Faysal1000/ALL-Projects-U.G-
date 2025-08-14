import aboutImage from "../assets/kahafil-ora-about-photo.png"; // Importing the about section image asset
import LogoAnimation from "./LogoAnimation";
import Collaboration from "./Collaboration";
const AboutSection = () => {
  return (
    // Section is a column so children can grow; min-h-screen keeps it at least viewport height
    <section className="w-full min-h-screen pt-0 flex flex-col">
      {/* Wrapper: fills section (flex-1), starts at top (justify-start), allows children to shrink (min-h-0) */}
      <div className="bg-[#fff] flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-[5.2%] min-h-0">
        {/* Top section */}
        <div className="flex flex-col items-center gap-[20px] self-stretch">
          <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-lg md:text-lg lg:text-xl font-[400] leading-normal uppercase">
            About Kahafil ora
          </div>

          <div className="text-black text-center font-['Poppins'] text-lg md:text-lg lg:text-xl font-[300] leading-normal capitalize">
            Kahafil Ora is an experienced IT Consultant known for delivering
            smart, tech-driven solutions to businesses. With a strong grasp of
            IT infrastructure and digital strategy, he helps organizations
            improve efficiency and achieve their goals through innovative
            technology
          </div>
        </div>

        {/* Divider */}
        <div className="self-stretch py-[4.2%]">
          <div className="h-0 border-t border-[#444]"></div>
        </div>

        {/* MIDDLE ROW — mobile: stacked (flex-col), desktop (md+) : row (original layout)
            Also, everything after the line starts from the left on mobile (items-start).
        */}
        <div className="flex flex-col md:flex-row w-full flex-1 items-start md:items-stretch justify-between min-h-0">
          {/* LEFT COLUMN — on mobile it becomes full width and left-aligned; on md+ it preserves original left alignment */}
          <div className="flex flex-col justify-between items-start flex-1 min-h-0 w-full md:w-auto">
            {/* Top block: Tagline + Roles (only takes content height) */}
            <div className="flex flex-col gap-2 w-full">
              <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-lg md:text-lg lg:text-xl font-[400] uppercase leading-normal text-left">
                Driving Digital Innovation Across Bangladesh and Beyond
              </div>

              <div className="flex flex-col gap-1 text-black font-['Poppins'] text-lg md:text-lg lg:text-xl font-light leading-[200%] tracking-[0.3px] capitalize text-left">
                <div>Mentor</div>
                <div>IT Strategist</div>
                <div>Tech Entrepreneur</div>
                <div>Cybersecurity Advocate</div>
              </div>
            </div>

            {/* Bottom block: Contact */}
            <div className="flex flex-col gap-2 pt-10 w-full">
              <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-lg md:text-lg lg:text-xl font-normal uppercase leading-normal text-left">
                Contact
              </div>
              <div className="text-black font-['Poppins'] text-lg md:text-lg lg:text-xl font-light leading-[200%] tracking-[0.4px] text-left">
                <div className="capitalize">kahafil Ora</div>
                <div className="lowercase">Dhaka | Bangladesh</div>
                <div className="lowercase">+880 1622-992222</div>
                <div className="lowercase">kahafil@goinnovior.com</div>
              </div>
            </div>
          </div>

          {/* CENTER IMAGE — centered on md+, on mobile it sits left with the rest (we keep it centered within its box) */}
          <div className="flex-shrink-0 flex items-center md:items-center justify-center md:justify-center w-full md:w-auto py-6 md:py-0">
            <div className="p-2 flex-shrink-0 rounded-[130px] border-[0.661px] border-[#444]">
              <div className="flex-shrink-0 rounded-[130px] border-none overflow-hidden">
                <img
                  src={aboutImage} // Imported image asset
                  alt="About"
                  className="rounded-[130px] 
                   w-[40vw] sm:w-[35vw] md:w-[23vw] lg:w-[25vw] xl:w-[16vw] 
                  min-w-[120px] max-w-[520px] aspect-[25/42] h-auto object-cover 
                   grayscale hover:grayscale-0 transition-all duration-800 ease-in-out transform hover:scale-[1.15]
                   mx-auto"
                />
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN SECTION
              - DESKTOP (md+): the original vertical right column, kept intact and visible only on md+ (hidden on mobile)
              - MOBILE (<md): a two-row layout (labels row, values row) that starts from left and spans full width
          */}

          {/* MOBILE: two-row headings+values (visible only on small screens) */}
          <div className="w-full md:hidden mt-4">
            {/* HEADINGS ROW */}
            <div className="grid grid-cols-3 gap-4 w-full text-left">
              <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-sm uppercase leading-normal">
                Years of Experience
              </div>
              <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-sm uppercase leading-normal">
                Satisfaction Clients
              </div>
              <div className="text-[rgba(68,68,68,0.5)] font-['Fragment_Mono'] text-sm uppercase leading-normal">
                CLIENTS ON WORLDWIDE
              </div>
            </div>

            {/* VALUES ROW */}
            <div className="grid grid-cols-3 gap-4 w-full mt-2">
              <div className="text-black font-['Poppins'] sm:text-3xl text-4xl font-light leading-normal">
                22+
              </div>
              <div className="text-black font-['Poppins'] sm:text-3xl text-4xl font-light leading-normal">
                100%
              </div>
              <div className="text-black font-['Poppins'] sm:text-3xl text-4xl font-light leading-normal">
                70+
              </div>
            </div>
          </div>

          {/* DESKTOP: original right column (visible from md and up) - unchanged content/layout */}
          <div className="hidden md:flex flex-col justify-between items-end flex-1 min-h-0 md:w-auto">
            <div className="flex flex-col items-end">
              <div className="text-[rgba(68,68,68,0.5)] text-right font-['Fragment_Mono'] text-sm lg:text-lg font-normal uppercase leading-normal">
                Years of Experience
              </div>
              <div className="text-black text-right font-['Poppins'] text-4xl md:text-5xl lg:text-6xl font-light leading-normal tracking-[0.8px] capitalize">
                22+
              </div>
            </div>

            <div className="flex flex-col items-end">
              <div className="text-[rgba(68,68,68,0.5)] text-right font-['Fragment_Mono'] text-sm lg:text-lg font-normal uppercase leading-normal">
                Satisfaction Clients
              </div>
              <div className="text-black text-right font-['Poppins'] text-4xl md:text-5xl lg:text-6xl font-light leading-normal tracking-[0.8px] capitalize">
                100%
              </div>
            </div>

            <div className="flex flex-col items-end">
              <div className="text-[rgba(68,68,68,0.5)] text-right font-['Fragment_Mono'] text-sm lg:text-lg font-normal uppercase leading-normal">
                CLIENTS ON WORLDWIDE
              </div>
              <div className="text-black text-right font-['Poppins'] text-4xl md:text-5xl lg:text-6xl font-light leading-normal tracking-[0.8px] capitalize">
                70+
              </div>
            </div>
          </div>
        </div>
      </div>
      <Collaboration />
      <LogoAnimation />
    </section>
  );
};

export default AboutSection;
