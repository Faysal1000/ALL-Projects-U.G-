import { motion } from "framer-motion";
import {
  FaInstagram,
  FaFacebookF,
  FaLinkedinIn,
  FaGlobe,
} from "react-icons/fa";
import { FiArrowLeft } from "react-icons/fi";
import FitToWidth from "./FitToWidth";
import { FaArrowTurnUp } from "react-icons/fa6";
import {
  FOOTER_IMAGES_AND_LINKS,
  SOCIAL_LINKS,
} from "/src/config/footerConfig";

const KAHAFIL_ORA_FONT_FAMILY = "'Poppins', 'Fragment Mono', sans-serif";

const iconForType = (type) => {
  switch (type) {
    case "instagram":
      return FaInstagram;
    case "facebook":
      return FaFacebookF;
    case "linkedin":
      return FaLinkedinIn;
    case "website":
      return FaGlobe;
    default:
      return null;
  }
};

const overlayVariants = {
  rest: { scale: 0, opacity: 0 },
  hover: { scale: 1.2, opacity: 1 },
};

const Footer = ({ items = FOOTER_IMAGES_AND_LINKS }) => {
  return (
    <footer className="bg-[#f8f8f8]">
      {/* image gallery */}
      <div className="flex h-[280px] items-start self-stretch overflow-x-auto md:overflow-hidden no-scrollbar">
        {items.map((item, idx) => {
          const Icon = iconForType(item.socialMediaType);
          return (
            <motion.div
              key={idx}
              className="relative flex-shrink-0 w-1/2 sm:w-1/3 md:w-1/6 h-full cursor-pointer overflow-hidden"
              initial="rest"
              whileHover="hover"
              animate="rest"
              onClick={() => {
                if (item.link)
                  window.open(item.link, "_blank", "noopener,noreferrer");
              }}
              role="link"
              aria-label={`Open ${item.socialMediaType || "link"}`}
            >
              <img
                src={item.image}
                alt={item.socialMediaType || `footer-image-${idx}`}
                className="w-full h-full object-cover block"
                draggable={false}
              />

              {Icon && (
                <motion.div
                  variants={overlayVariants}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="pointer-events-none absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-28 h-28 flex items-center justify-center"
                  style={{
                    backdropFilter: "blur(5px)",
                    background: "rgba(255, 255, 255, 0.10)",
                    opacity: 0.9,
                  }}
                >
                  <div className="relative flex items-center justify-center">
                    <div className="relative z-10">
                      <Icon className="text-white text-5xl" />
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>
      {/* ===================== KAHAFIL ORA RESPONSIVE TEXT ===================== */}
      {/* Shared styling applied at parent level so children don't repeat styles */}
      <div
        className="w-full text-[#444] uppercase font-[700]"
        style={{
          fontFamily: KAHAFIL_ORA_FONT_FAMILY,
          letterSpacing: "0.02em",
          WebkitTextStrokeWidth: "5px",
          WebkitTextStrokeColor: "#444",
        }}
      >
        {/* ===== Desktop full view: "Kahafil Ora" (unchanged) ===== */}
        <div className="hidden md:flex md:flex-col md:items-start md:w-full">
          <FitToWidth
            minFont={18}
            maxFont={700}
            precision={0.3}
            className="w-full pt-2"
            style={{
              minHeight: 0,
              lineHeight: 1,
              margin: 0,
              padding: 0,
            }}
            /* keep default FitToWidth behaviour here */
          >
            Kahafil Ora
          </FitToWidth>
        </div>

        {/*** MOBILE: Two rows with NO GAP between them ***/
        /* Key points:
              - container has no gap/padding/margin (gap-0, p-0, m-0)
              - FitToWidth is forced to minHeight:0 and tight lineHeight
              - Ora block is right-aligned (w-1/2 ml-auto) and has no top margin
          */}
        <div
          className="md:hidden w-full flex flex-col gap-0 p-0 m-0"
          /* ensure no extra spacing from parent */
        >
          {/* Row 1: Kahafil (full width, no extra spacing) */}
          <div className="w-full p-0 m-0">
            <FitToWidth
              minFont={18}
              maxFont={700}
              precision={0.3}
              className="w-full"
              style={{
                minHeight: 0,
                lineHeight: 1,
                margin: 0,
                padding: 0,
              }}
            >
              Kahafil
            </FitToWidth>
          </div>

          {/* Row 2: Ora (immediately after Kahafil, right-aligned, no gap) */}
          <div className="w-full flex p-0 m-0 mt-0">
            <div
              className="w-1/2 ml-auto flex justify-end items-start p-0 m-0"
              /* items-start keeps "ora" aligned to top of this half (no extra spacing) */
            >
              <FitToWidth
                minFont={18}
                maxFont={700}
                precision={0.3}
                className="w-full text-right"
                style={{
                  minHeight: 0,
                  lineHeight: 1,
                  margin: 0,
                  padding: 0,
                }}
              >
                ora
              </FitToWidth>
            </div>
          </div>
        </div>
      </div>
      {/* ===================== END KAHAFIL ORA RESPONSIVE TEXT ===================== */}

      <div className="w-full pt-10 px-10 text-xl 2xl:text-2xl font-[Poppins,sans-serif] text-[#444] capitalize">
        {/*
        Mobile: vertical stack with nice gap.
        Desktop (md+): single row, no wrap, equal spacing between columns via justify-between.
        Important: we remove md:gap to let justify-between control spacing perfectly.
      */}
        <div className="flex flex-col gap-6 md:flex-row md:flex-nowrap md:items-start md:justify-between md:gap-0">
          {/* ==== Column 1 — Person info (left aligned) ==== */}
          <div className="shrink-0 text-left">
            <div className="flex flex-col space-y-2">
              {/* constant border, color fades on hover */}
              <span className="inline-block font-light capitalize pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out">
                Kahafil@goinnovior.com
              </span>
              <span className="inline-block pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out">
                +880 1622-992222
              </span>
            </div>
          </div>

          {/* ==== Column 2 — Company info (arrow only animation) ==== */}
          <div className="shrink-0 text-left">
            <dl className="flex flex-col space-y-2">
              <div className="group">
                <dd className="font-light capitalize flex items-center transition-all duration-500">
                  <span className="inline-flex items-center w-0 group-hover:w-5 overflow-hidden transition-all duration-500 mr-0 group-hover:mr-2">
                    <FiArrowLeft />
                  </span>
                  <span className="inline-block pb-0.5 w-fit">
                    Goinnovior Limited
                  </span>
                </dd>
              </div>
              <div className="group">
                <dd className="flex items-center transition-all duration-500">
                  <span className="inline-flex items-center w-0 group-hover:w-5 overflow-hidden transition-all duration-500 mr-0 group-hover:mr-2">
                    <FiArrowLeft />
                  </span>
                  <span className="inline-block pb-0.5 w-fit">
                    360d soul limited
                  </span>
                </dd>
              </div>
              <div className="group">
                <dd className="flex items-center transition-all duration-500">
                  <span className="inline-flex items-center w-0 group-hover:w-5 overflow-hidden transition-all duration-500 mr-0 group-hover:mr-2">
                    <FiArrowLeft />
                  </span>
                  <span className="inline-block pb-0.5 w-fit">codinnovior</span>
                </dd>
              </div>
              <div className="group">
                <dd className="flex items-center transition-all duration-500">
                  <span className="inline-flex items-center w-0 group-hover:w-5 overflow-hidden transition-all duration-500 mr-0 group-hover:mr-2">
                    <FiArrowLeft />
                  </span>
                  <span className="inline-block pb-0.5 w-fit">
                    lifeinnovior
                  </span>
                </dd>
              </div>
            </dl>
          </div>

          {/* ==== Column 3 — Address (left aligned) ==== */}
          <div className="shrink-0 text-left">
            <div className="flex flex-col space-y-2">
              <span className="inline-block font-light capitalize pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out">
                Lift 4, House 774, Road 11, Avenue
              </span>
              <span className="inline-block pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out">
                Mirpur DOHS
              </span>
              <span className="inline-block pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out">
                Dhaka 1216
              </span>
              <span className="inline-block pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out">
                Bangladesh
              </span>
            </div>
          </div>

          {/* ==== Column 4 — Social links ==== */}
          <div className="shrink-0 text-left">
            <nav
              aria-label="social links"
              className="flex flex-col space-y-2 items-start"
            >
              {SOCIAL_LINKS.map(({ label, href }) => (
                <a
                  key={label}
                  href={href}
                  target={href.startsWith("http") ? "_blank" : undefined}
                  rel={
                    href.startsWith("http") ? "noopener noreferrer" : undefined
                  }
                  className="cursor-pointer inline-block pb-0.5 w-fit border-b border-transparent hover:border-black/50 transition-colors duration-500 ease-in-out hover:text-indigo-600"
                >
                  {label}
                </a>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {/* Last portion */}
      <div className="text-[#444] font-poppins text-xl font-light leading-normal capitalize flex w-full px-[60px] mt-10 py-[30px] justify-between bg-[#EBEBEB] flex-col md:flex-row items-start md:items-center space-y-4 md:space-y-0">
        {/* Arrow first on mobile */}
        <div className="order-first md:order-last">
          <FaArrowTurnUp size={20} />
        </div>
        <div>Copyright © 2025 Kahafil Ora</div>
        <div>Privacy & Policy</div>
        <div>Terms & Conditions</div>
      </div>
    </footer>
  );
};

export default Footer;
