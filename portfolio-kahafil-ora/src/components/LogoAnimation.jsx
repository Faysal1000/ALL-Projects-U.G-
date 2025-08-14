import "../index.css"; // CSS for slide-in animation
import {
  FaFacebook,
  FaInstagram,
  FaTwitter,
  FaLinkedin,
  FaYoutube,
  FaGithub,
  FaWhatsapp,
} from "react-icons/fa";
import { GiMailbox } from "react-icons/gi";
import { SiGmail, SiTiktok, SiDiscord, SiPinterest } from "react-icons/si";

// Configurable logos array with icons
const logos = [
  { name: "Gmail", icon: <SiGmail size={40} /> },
  { name: "Facebook", icon: <FaFacebook size={40} /> },
  { name: "Instagram", icon: <FaInstagram size={40} /> },
  { name: "Twitter", icon: <FaTwitter size={40} /> },
  { name: "LinkedIn", icon: <FaLinkedin size={40} /> },
  { name: "YouTube", icon: <FaYoutube size={40} /> },
  { name: "GitHub", icon: <FaGithub size={40} /> },
  { name: "WhatsApp", icon: <FaWhatsapp size={40} /> },
  { name: "TikTok", icon: <SiTiktok size={40} /> },
  { name: "Discord", icon: <SiDiscord size={40} /> },
  { name: "Pinterest", icon: <SiPinterest size={40} /> },
  { name: "Mailbox", icon: <GiMailbox size={40} /> },
];

// Logo animation component
const LogoAnimation = () => {
  return (
    <section className="bg-[#fff] flex-1 flex py-[3%] px-[12.5%] flex-col justify-start items-start gap-[105.2%] min-h-0">
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-semibold leading-normal mb-8 md:mb-10">
        I help brands to drive results
        <span className="text-[#9747FF]">.</span> {/* different color '.' */}
      </div>

      {/* ---------------------------
          LOGO MARQUEE (two directions)
          - Animation & mask live in CSS (index.css)
          - Tailwind controls spacing & responsive sizing
          - Padding on the container keeps logos away from the mask edges so they aren't cut
          - min-w on each item ensures stable widths (no shrinking or reflow during animation)
          ----------------------------*/}
      {/* LEFT → RIGHT  */}
      <div
        className="w-full logo-container px-4 md:px-8 lg:px-12 py-4"
        aria-hidden="true"
      >
        <div
          className="logo-track"
          data-direction="left"
          style={{ "--speed": "18s", "--copies": 3 }}
        >
          {[...logos, ...logos, ...logos].map((logo, index) => (
            <div
              key={index}
              className="flex items-center gap-3 px-3 py-2 text-sm md:text-lg min-w-[110px] sm:min-w-[130px] md:min-w-[170px]"
            >
              <div className="flex items-center justify-center flex-shrink-0 w-10 h-6 sm:w-12 sm:h-7 md:w-20 md:h-10 lg:w-24 lg:h-12">
                {logo.icon}
              </div>
              <div className="text-[#444] font-['Fragment_Mono'] font-normal uppercase text-xs sm:text-sm md:text-base lg:text-lg">
                {logo.name}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT → LEFT */}
      <div
        className="w-full logo-container px-4 md:px-8 lg:px-12 py-4"
        aria-hidden="true"
      >
        <div
          className="logo-track"
          data-direction="right"
          style={{ "--speed": "18s", "--copies": 3 }}
        >
          {[...logos, ...logos, ...logos].map((logo, index) => (
            <div
              key={index}
              className="flex items-center gap-3 px-3 py-2 text-sm md:text-lg min-w-[110px] sm:min-w-[130px] md:min-w-[170px]"
            >
              <div className="flex items-center justify-center flex-shrink-0 w-10 h-6 sm:w-12 sm:h-7 md:w-20 md:h-10 lg:w-24 lg:h-12">
                {logo.icon}
              </div>
              <div className="text-[#444] font-['Fragment_Mono'] font-normal uppercase text-xs sm:text-sm md:text-base lg:text-lg">
                {logo.name}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default LogoAnimation;
