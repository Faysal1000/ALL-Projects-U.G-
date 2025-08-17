import heroImage from "../assets/kahafil_ora_about_hero_image.png";
import kahafiloraimage from "../assets/kahafil-ora-image.png";

const AboutUsHeroSection = () => {
  return (
    // height uses calc(100vh - var(--navbar-height, 0px)).
    // If your navbar sets --navbar-height (eg. :root { --navbar-height: 64px }),
    // that value will be subtracted. Otherwise section becomes full 100vh.
    <section
      className="self-stretch mt-15 flex relative overflow-hidden"
      style={{ height: "calc(100vh - 64px)" }}
    >
      {/* LEFT COLUMN (grayscale + portrait + badge + quote) */}
      <div className="w-[35.0%] flex-shrink-0 relative h-full">
        <div className="flex flex-col justify-between items-start flex-shrink-0 relative overflow-visible h-full">
          {/* Grayscale background (fills left column and contributes to height) */}
          <div
            className="absolute inset-0 grayscale -z-10"
            style={{
              backgroundImage: `url(${heroImage})`,
              backgroundRepeat: "no-repeat",
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          />

          {/* Name Badge */}
          <div className="flex px-6 py-3 justify-center items-center gap-2 rounded bg-[rgba(0,0,0,0.20)] backdrop-blur-[10px]">
            <p className="text-white font-[Poppins] text-sm font-bold uppercase leading-normal">
              kahafil ora
            </p>
          </div>

          {/* Quote Section */}
          <div className="flex justify-center items-center gap-[10px] self-stretch px-[20px] py-[117px] pl-35">
            <div className="flex-1 text-white font-poppins text-[12px] italic font-light leading-normal capitalize">
              I believe in progress driven by innovation and integrity. Every
              project is an opportunity to create real impact—with purpose,
              clarity, and care. This portfolio reflects not just what we’ve
              built, but how we think, lead, and grow.
            </div>
          </div>

          {/* Decorative portrait (absolute or in-flow is fine; this stays as you had it) */}
          <div
            className="aspect-[3/4] h-1/3 flex-shrink-0 absolute top-1/3 right-[-60px] -translate-y-1/2 bg-no-repeat z-20"
            style={{
              backgroundImage: `url(${kahafiloraimage})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          />
        </div>
      </div>

      {/* RIGHT COLUMN
          - h-full so it matches the section height
          - overflow-auto so content scrolls inside the column instead of extending page height
      */}
      <div className="flex flex-col items-start gap-10 flex-shrink-0 w-[65.0%] ml-10 px-[60px] pt-[45px] h-full overflow-auto">
        {/* === Block 1 (top row): left menu + biography text + scroll === */}
        <div className="flex w-full min-h-1/3 items-start flex-shrink-0">
          {/* Left-side vertical menu + scroll indicator */}
          <div className="flex flex-col justify-between items-start self-stretch pr-5">
            <div className="flex flex-col items-start gap-3.5">
              <div className="flex items-center gap-2.5 border-b border-transparent hover:border-gray-700">
                <p className="text-[rgba(68,68,68,0.5)] font-[Poppins] text-sm font-normal leading-normal uppercase">
                  Biography
                </p>
              </div>
              <div className="flex items-center gap-2.5 border-b border-transparent hover:border-gray-700">
                <p className="text-[rgba(68,68,68,0.5)] font-[Poppins] text-sm font-normal leading-normal uppercase">
                  Vision & Values
                </p>
              </div>
              <div className="flex items-center gap-2.5 border-b border-transparent hover:border-gray-700">
                <p className="text-[rgba(68,68,68,0.5)] font-[Poppins] text-sm font-normal leading-normal uppercase">
                  Words I Live By
                </p>
              </div>
              <div className="flex items-center gap-2.5 border-b border-transparent hover:border-gray-700">
                <p className="text-[rgba(68,68,68,0.5)] font-[Poppins] text-sm font-normal leading-normal uppercase">
                  Leadership Journey
                </p>
              </div>
              <div className="flex items-center gap-2.5 border-b border-transparent hover:border-gray-700">
                <p className="text-[rgba(68,68,68,0.5)] font-[Poppins] text-sm font-normal leading-normal uppercase">
                  My Story
                </p>
              </div>
            </div>

            {/* scroll down */}
            <div className="flex flex-col items-start">
              <p className="text-[rgba(68,68,68,0.6)] font-[Poppins] text-sm uppercase mb-0">
                Scroll Down
              </p>

              {/* Arrow: line + tip */}
              <div className="flex flex-col items-center animate-stretchArrow origin-top">
                <div className="w-[1px] h-[40px] bg-[rgba(68,68,68,0.6)]" />
                <div className="w-3 h-3 border-r-2 border-b-2 border-[rgba(68,68,68,0.6)] rotate-45 -mt-3" />
              </div>

              <style jsx>{`
                @keyframes stretchArrow {
                  0%,
                  100% {
                    transform: scaleY(0.6);
                  }
                  50% {
                    transform: scaleY(2);
                  }
                }
                .animate-stretchArrow {
                  animation: stretchArrow 2.5s ease-in-out infinite;
                }
              `}</style>
            </div>
          </div>

          {/* Right-side biography text (fills remaining width) */}
          <div className="flex-1 m-auto min-w-0">
            <div className="text-sm text-[rgba(68,68,68,0.9)] font-[Poppins] leading-relaxed">
              <p className="mb-4">
                A visionary leader, with over 22 years of experience, Kahafil
                Ora has consistently championed innovation, growth, and
                meaningful change. As the Managing Director of Goinnovior
                Limited and co-founder of 360D Soul, Lifeinnovior, and
                Codeinnovior, he has successfully built ventures that bridge
                technology, business, and social development. His journey
                reflects a rare blend of entrepreneurial grit and a deep
                commitment to community empowerment.
              </p>

              <p className="mb-4">
                Under his leadership, Goinnovior Limited has emerged as a
                reliable provider of ICT and information security solutions —
                supporting businesses with future-ready technology. Meanwhile,
                through Lifeinnovior, he advocates for mental health awareness
                and care, addressing a vital but often overlooked area of
                well-being. His involvement in 360D Soul and Codeinnovior
                highlights his passion for knowledge-sharing and youth
                empowerment, creating platforms where free and accessible tech
                education is a reality.
              </p>

              <p className="mb-4">
                Kahafil Ora believes that leadership is not just about driving
                success—it's about creating sustainable impact. He envisions a
                future where technology uplifts people, not just industries. The
                projects and initiatives he leads aim to build a legacy that
                inspires purpose, compassion, and continuous learning.
              </p>
            </div>
          </div>
        </div>

        {/* === Block 2 (highlight + 2 images) - stays at bottom with mt-auto */}
        <div className="w-full mt-auto">
          <div className="flex w-full items-start gap-8">
            {/* Highlight paragraph */}
            <div className="w-1/3 px-2">
              <p className="text-[14px] text-[rgba(68,68,68,0.9)] leading-relaxed">
                Kahafil Ora was honored with a crest upon successfully achieving
                the{" "}
                <strong>
                  Advanced Certificate For Management Professionals 4.0 (ACMP
                  4.0)
                </strong>{" "}
                from IBA, University of Dhaka.
              </p>
            </div>

            {/* Two images */}
            <div className="w-2/3 flex flex-col gap-4">
              <div className="flex w-full items-center justify-between gap-4">
                <div
                  className="w-[49%] h-[140px] bg-no-repeat bg-cover"
                  style={{
                    backgroundImage: `url(${kahafiloraimage})`,
                    backgroundPosition: "center",
                  }}
                />
                <div
                  className="w-[49%] h-[140px] bg-no-repeat bg-cover"
                  style={{
                    backgroundImage: `url(${kahafiloraimage})`,
                    backgroundPosition: "center",
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutUsHeroSection;
