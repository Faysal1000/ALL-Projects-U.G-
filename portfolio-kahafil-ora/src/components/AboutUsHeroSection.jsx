import { useState } from "react";
import heroImage from "../assets/kahafil_ora_about_hero_image.png";
import kahafiloraimage from "../assets/kahafilOra_aboutus_image.jpg";
import aboutUsThumbnailimage1 from "../assets/aboutUsThumbnailimage1.jpg";
import aboutUsThumbnailimage2 from "../assets/aboutUsThumbnailimage2.png";

/**
 * AboutUsHeroSection component:
 * - Desktop layout: split into two columns (35% left hero, 65% right content).
 * - Mobile layout: single stacked column, left area turns into a hero banner.
 * - Features: name badge, inspirational quote, portrait (desktop only), tabs for content, scroll indicator, highlight section, and images.
 */

const AboutUsHeroSection = () => {
  // Menu tabs
  const tabs = [
    { key: "biography", label: "Biography" },
    { key: "vision", label: "Vision & Values" },
    { key: "words", label: "Words I Live By" },
    { key: "journey", label: "Leadership Journey" },
    { key: "story", label: "My Story" },
  ];

  // State for selected tab
  const [displayTab, setDisplayTab] = useState("biography");
  // Animation state for fade-in/out when switching tabs
  const [animating, setAnimating] = useState(false);

  // Handles tab click (with small delay for animation)
  const handleTabClick = (key) => {
    if (key === displayTab || animating) return;
    setAnimating(true);
    setTimeout(() => {
      setDisplayTab(key);
      setTimeout(() => setAnimating(false), 20);
    }, 180);
  };

  return (
    // Main wrapper section: responsive, two-column on desktop, stacked on mobile
    <section
      className="self-stretch mt-15 flex flex-col lg:flex-row relative overflow-hidden"
      style={{ minHeight: "calc(100vh - 64px)" }}
      aria-label="About Kahafil Ora"
    >
      {/* =========================
          MOBILE HERO (mobile only)
          - 65vh tall hero banner
          - Shows background image
          - Includes badge and bottom quote
          - No portrait on mobile
      ========================= */}
      <div className="block lg:hidden w-full flex-shrink-0 relative h-[65vh] overflow-hidden">
        {/* Hero background image */}
        <div
          className="absolute inset-0 -z-10 bg-no-repeat bg-center bg-cover"
          style={{
            backgroundImage: `url(${heroImage})`,
            backgroundSize: "cover",
          }}
          aria-hidden="true"
        />

        {/* Top-left name badge */}
        <div className="absolute z-20">
          <div className="px-4 py-2 rounded bg-[rgba(0,0,0,0.20)] backdrop-blur-[10px]">
            <p className="text-white font-[Poppins] text-sm font-bold uppercase leading-normal">
              kahafil ora
            </p>
          </div>
        </div>

        {/* Quote anchored at bottom center */}
        <div className="absolute left-0 right-0 bottom-4 z-20 px-6">
          <div className="text-white font-poppins text-base italic font-light leading-relaxed text-center backdrop-blur-[2px]">
            I believe in progress driven by innovation and integrity. Every
            project is an opportunity to create real impact—with purpose,
            clarity, and care. This portfolio reflects not just what we’ve
            built, but how we think, lead, and grow.
          </div>
        </div>
      </div>

      {/* ===========================
          LEFT COLUMN (desktop only)
          - 35% width hero section
          - Grayscale background
          - Name badge
          - Quote text block
          - Absolute positioned portrait
      =========================== */}
      <div className="hidden lg:block w-[35.0%] flex-shrink-0 relative h-full">
        <div className="flex flex-col justify-between items-start flex-shrink-0 relative overflow-visible md:min-h-[calc(100vh-64px)]">
          {/* Grayscale hero background */}
          <div
            className="absolute inset-0 grayscale -z-10"
            style={{
              backgroundImage: `url(${heroImage})`,
              backgroundRepeat: "no-repeat",
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          />

          {/* Name badge */}
          <div className="flex px-4 py-2 justify-center items-center gap-2 rounded bg-[rgba(0,0,0,0.20)] backdrop-blur-[10px]">
            <p className="text-white font-[Poppins] text-sm font-bold uppercase leading-normal">
              kahafil ora
            </p>
          </div>

          {/* Quote text (desktop layout) */}
          <div className="flex justify-center items-center gap-[10px] self-stretch px-[20px] py-[117px] pl-35">
            <div className="flex-1 text-white font-poppins text-[12px] italic font-light leading-normal capitalize">
              I believe in progress driven by innovation and integrity. Every
              project is an opportunity to create real impact—with purpose,
              clarity, and care. This portfolio reflects not just what we’ve
              built, but how we think, lead, and grow.
            </div>
          </div>

          {/* Decorative portrait (desktop only, absolute positioned) */}
          <div
            className="aspect-[3/4] h-1/3 flex-shrink-0 absolute top-1/4 right-[-60px] -translate-y-1/2 bg-no-repeat z-20"
            style={{
              backgroundImage: `url(${kahafiloraimage})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          />
        </div>
      </div>

      {/* ===========================
          RIGHT COLUMN
          - Contains menu (tabs), content, scroll indicator, highlight, images
          - Full width on mobile, 65% on desktop
      =========================== */}
      <div className="flex flex-col items-start gap-10 flex-shrink-0 w-full lg:w-[65.0%] ml-0 lg:ml-10 px-6 lg:px-[60px] pt-6 lg:pt-[45px] h-auto lg:h-full overflow-visible lg:overflow-auto">
        {/* === Block 1: menu (tabs) + content area === */}
        <div className="flex w-full flex-col lg:flex-row min-h-1/3 items-start flex-shrink-0">
          {/* Tabs menu */}
          <div className="flex flex-col justify-between items-start self-stretch pr-0 lg:pr-5 w-full lg:w-auto">
            <div className="flex flex-col items-start gap-1.5 md:gap-3.5 w-full lg:w-auto">
              {/* Each tab button */}
              {tabs.map((t) => {
                const isActive = displayTab === t.key;
                return (
                  <div
                    key={t.key}
                    className="w-full lg:w-auto"
                    role="button"
                    tabIndex={0}
                    onClick={() => handleTabClick(t.key)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ")
                        handleTabClick(t.key);
                    }}
                  >
                    <span
                      className={`inline-block pb-1 cursor-pointer transition-colors duration-600 font-[Poppins] text-sm uppercase ${
                        isActive
                          ? "border-b-2 border-gray-800/50 text-[rgba(68,68,68,0.95)] font-semibold"
                          : "border-b-2 border-transparent text-[rgba(68,68,68,0.5)] hover:border-gray-700 hover:text-[rgba(68,68,68,0.9)]"
                      }`}
                    >
                      {t.label}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Scroll down indicator (desktop only) */}
            <div className="hidden lg:flex lg:flex-col lg:items-start lg:mt-6">
              <p className="text-[rgba(68,68,68,0.6)] font-[Poppins] text-sm uppercase mb-0">
                Scroll Down
              </p>

              {/* Animated vertical line + arrow */}
              <div className="flex flex-col items-center animate-stretchArrow origin-top mt-1">
                <div className="w-[1px] h-[40px] bg-[rgba(68,68,68,0.6)]" />
                <div className="w-3 h-3 border-r-2 border-b-2 border-[rgba(68,68,68,0.6)] rotate-45 -mt-3" />
              </div>

              {/* Keyframe animation for stretch effect */}
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
                  animation: stretchArrow 3s ease-in-out infinite;
                }
              `}</style>
            </div>
          </div>

          {/* Tab content area */}
          <div className="flex-1 m-auto lg:min-h-[48vh] min-w-0 mt-6 lg:mt-0">
            <div
              className={`transition-opacity duration-180 ${animating ? "opacity-0" : "opacity-100"}`}
            >
              {/* Biography tab content */}
              {displayTab === "biography" && (
                <div className="text-sm text-[rgba(68,68,68,0.9)] font-[Poppins] leading-relaxed">
                  <p className="mb-4">
                    A visionary leader, with over 22 years of experience,
                    Kahafil Ora has consistently championed innovation, growth,
                    and meaningful change. As the Managing Director of
                    Goinnovior Limited and co-founder of 360D Soul,
                    Lifeinnovior, and Codeinnovior, he has successfully built
                    ventures that bridge technology, business, and social
                    development. His journey reflects a rare blend of
                    entrepreneurial grit and a deep commitment to community
                    empowerment.
                  </p>

                  <p className="mb-4">
                    Under his leadership, Goinnovior Limited has emerged as a
                    reliable provider of ICT and information security solutions
                    — supporting businesses with future-ready technology.
                    Meanwhile, through Lifeinnovior, he advocates for mental
                    health awareness and care, addressing a vital but often
                    overlooked area of well-being. His involvement in 360D Soul
                    and Codeinnovior highlights his passion for
                    knowledge-sharing and youth empowerment, creating platforms
                    where free and accessible tech education is a reality.
                  </p>

                  <p className="mb-4">
                    Kahafil Ora believes that leadership is not just about
                    driving success—it's about creating sustainable impact. He
                    envisions a future where technology uplifts people, not just
                    industries. The projects and initiatives he leads aim to
                    build a legacy that inspires purpose, compassion, and
                    continuous learning.
                  </p>
                </div>
              )}

              {/* Vision tab */}
              {displayTab === "vision" && (
                <div className="text-sm text-[rgba(68,68,68,0.9)] font-[Poppins] leading-relaxed lg:pl-[18%]">
                  <p>
                    <strong>Vision:</strong>
                  </p>
                  {/* Vision statement */}
                  <p className="mb-4">
                    To build a future where technology, leadership, and
                    compassion come together to create sustainable impact—for
                    businesses, individuals, and society as a whole. I believe
                    true success lies not only in innovation but in the ability
                    to uplift and empower those around us.
                  </p>

                  {/* Core Values heading */}
                  <p className="font-semibold mb-0">Core Values</p>

                  {/* Core Values list */}
                  <ul className="list-disc pl-5">
                    <li>
                      <strong>Integrity</strong> – Every action and decision is
                      rooted in honesty, ethics, and accountability.
                    </li>
                    <li>
                      <strong>Innovation with Purpose</strong> – I pursue
                      technology not just for progress, but for positive,
                      meaningful change.
                    </li>
                    <li>
                      <strong>People First</strong> – Whether in business or
                      community, people are at the heart of every solution I
                      design or support.
                    </li>
                    <li>
                      <strong>Continuous Learning</strong> – Growth never stops.
                      I embrace change, challenge norms, and seek to evolve.
                    </li>
                    <li>
                      <strong>Empowerment</strong> – I am committed to creating
                      opportunities—especially through education, mental
                      well-being, and digital access—for those who need it most.
                    </li>
                  </ul>
                </div>
              )}

              {/* Words I Live By tab */}
              {displayTab === "words" && (
                <div className="text-sm text-[rgba(68,68,68,0.9)] font-[Poppins] leading-relaxed lg:pl-[18%]">
                  <p className="mb-5">
                    "Leadership is not about being in charge. It’s about taking
                    care of those in your charge."
                    <br />– <strong>Simon Sinek</strong>
                  </p>

                  <p className="mb-5">
                    "Innovation is seeing what everybody has seen and thinking
                    what nobody has thought."
                    <br />– <strong>Dr. Albert Szent-Györgyi</strong>
                  </p>

                  <p className="mb-5">
                    "Success is not measured by what you have, but by the
                    positive impact you leave behind."
                    <br />– <strong>Kahafil Ora</strong>
                  </p>
                </div>
              )}

              {/* Journey tab */}
              {displayTab === "journey" && (
                <div className="text-sm text-[rgba(68,68,68,0.9)] font-[Poppins] leading-relaxed lg:pl-[18%]">
                  <p className="mb-4">
                    With over 22 years of experience across technology,
                    business, and social innovation, my leadership journey has
                    always been guided by one principle: create impact with
                    intention.
                  </p>

                  <p className="mb-4">
                    It began with a passion for IT solutions, evolving into a
                    broader mission—founding and leading multiple ventures like{" "}
                    <strong>Goinnovior Limited</strong>,{" "}
                    <strong>360D Soul Ltd.</strong>,{" "}
                    <strong>Lifeinnovior</strong>, and{" "}
                    <strong>Codeinnovior</strong>. Each of these was born from a
                    real-world need: transforming businesses through ICT and
                    InfoSec solutions, promoting mental health awareness, and
                    offering free tech education to the underserved.
                  </p>

                  <p className="mb-4">
                    Through every challenge and milestone, I’ve stayed grounded
                    in continuous learning and human connection. Whether it’s
                    earning advanced management credentials from{" "}
                    <strong>IBA, University of Dhaka</strong>, or mentoring the
                    next generation, leadership for me is not about a title—it’s
                    about responsibility, growth, and purpose.
                  </p>
                </div>
              )}

              {/* Story tab */}
              {displayTab === "story" && (
                <div className="text-sm text-[rgba(68,68,68,0.9)] font-[Poppins] leading-relaxed lg:pl-[18%]">
                  <p className="mb-4">
                    I come from a place where dreams were simple but ambitions
                    were strong. My journey began with a deep curiosity about
                    technology and a desire to solve problems that mattered.
                    Over the years, that curiosity turned into a mission—to
                    create meaningful change through innovation, leadership, and
                    service.
                  </p>

                  <p className="mb-4">
                    From humble beginnings to becoming the Managing Director of{" "}
                    <strong>Goinnovior Limited</strong> and Co-founder of{" "}
                    <strong>360D Soul</strong>, <strong>Lifeinnovior</strong>,
                    and <strong>Codeinnovior</strong>, every step of the way has
                    been driven by purpose. I've witnessed how technology can
                    transform businesses—but more importantly, how it can uplift
                    people. That belief led me to support mental health
                    initiatives, advocate for youth development, and promote
                    free tech education for those who need it most.
                  </p>

                  <p className="mb-4">
                    This story is still being written. Every project, every
                    challenge, every person I meet continues to shape who I am.
                    I don’t just lead companies—I build communities, nurture
                    ideas, and stay committed to growth that goes beyond the
                    bottom line. Because at the heart of it all, I believe in
                    building a better future—for everyone.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* === Block 2: highlight section + two images side-by-side === */}
        <div className="w-full mt-0 lg:mt-auto">
          <div className="flex flex-col lg:flex-row w-full items-center gap-4 lg:gap-8">
            {/* Highlight text block */}
            <div className="w-full lg:w-1/3 px-0 lg:px-2">
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

            {/* Two thumbnail images */}
            <div className="w-full lg:w-2/3 flex flex-col items-end sm:flex-row gap-4 min-h-[180px]">
              <div
                className="w-full aspect-[16/9] sm:w-[49%] h-auto bg-no-repeat bg-cover bg-center rounded-sm"
                style={{ backgroundImage: `url(${aboutUsThumbnailimage1})` }}
                aria-hidden="true"
              />
              <div
                className="w-full aspect-[16/9] sm:w-[49%] h-auto bg-no-repeat bg-cover bg-center rounded-sm"
                style={{ backgroundImage: `url(${aboutUsThumbnailimage2})` }}
                aria-hidden="true"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutUsHeroSection;
