import React, { useState } from "react";
import {
  FaFacebookF,
  FaTwitter,
  FaBars,
  FaTimes,
  FaLinkedin,
} from "react-icons/fa";
import { GrMail } from "react-icons/gr";
//import texture from "../assets/texture.svg";

// ======= Configurable Variables =======
const TEXTURE= 'https://www.transparenttextures.com/patterns/asfalt-dark.png';
const TEXT_COLOR = "#444";
const HOVER_TEXT_COLOR = "#4a4a4a";
const UNDERLINE_COLOR = "#444";
const MOBILE_BG = "#EFEAE4";
const FONT_FAMILY = "'Fragment Mono', sans-serif";

// ======= Nav Items =======
const navItems = [
  { name: "WORK", href: "/" },
  { name: "ABOUT", href: "/" },
  { name: "THOUGHTS", href: "/" },
];

// ======= Social Links =======
const socialLinks = [
  { name: "MAIL", href: "#", icon: <GrMail size={20} /> },
  { name: "X", href: "#", icon: <FaTwitter size={20} /> },
  { name: "FB", href: "#", icon: <FaFacebookF size={20} /> },
  { name: "LI", href: "#", icon: <FaLinkedin size={20} /> },
];

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const toggleMenu = () => setIsOpen(!isOpen);

  return (

      <nav
        className="bg-cover bg-no-repeat sticky top-0 z-50"
        style={{
          backgroundImage: `url(${TEXTURE})`
        }}
      >
      <div className="container mx-auto py-2 flex items-center">
        {/* Desktop Menu */}
        <div className="hidden md:flex flex-[1] justify-between">
          {navItems.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="relative text-lg font group px-2 py-2 transition-colors duration-300"
              style={{ color: TEXT_COLOR }}
            >
              {item.name}
             <span
                className="absolute bottom-2 left-1/2 w-full h-px transform -translate-x-1/2 scale-x-0 origin-center transition-transform duration-300 ease-out group-hover:scale-x-100"
                style={{ backgroundColor: UNDERLINE_COLOR, fontFamily: FONT_FAMILY }}
              ></span>
            </a>
          ))}

          {/* Desktop Social Links */}
          <div className="hidden md:flex justify-between w-[20%]">
            {socialLinks.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="relative text-lg group px-2 py-2 transition-colors duration-300"
                style={{ color: TEXT_COLOR }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = HOVER_TEXT_COLOR)
                }
                onMouseLeave={(e) => (e.currentTarget.style.color = TEXT_COLOR)}
              >
                {item.name}

                <span
                  className="absolute bottom-2 left-1/2 w-full h-px transform -translate-x-1/2 scale-x-0 origin-center transition-transform duration-300 ease-out group-hover:scale-x-100"
                  style={{ backgroundColor: UNDERLINE_COLOR, fontFamily: FONT_FAMILY }}
                ></span>
              </a>
            ))}
          </div>
        </div>

        {/* Mobile Hamburger Icon */}
        <div className="md:hidden ml-auto">
          <button
            onClick={toggleMenu}
            style={{ color: TEXT_COLOR, fontFamily: FONT_FAMILY }}
            className="focus:outline-none"
          >
            {isOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu (Left Sidebar) */}
      <div
        className={`md:hidden fixed top-0 left-0 h-full w-64 shadow-xl transform transition-transform duration-300 ease-in-out z-40 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{ backgroundColor: MOBILE_BG }}
      >
        <div className="flex flex-col p-6 space-y-6 mt-16">
          {navItems.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="text-lg font-medium transition-colors duration-300"
              style={{ color: TEXT_COLOR }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.color = HOVER_TEXT_COLOR)
              }
              onMouseLeave={(e) => (e.currentTarget.style.color = TEXT_COLOR)}
              onClick={toggleMenu}
            >
              {item.name}
            </a>
          ))}

          {/* Mobile Social Links */}
          <div className="flex justify-evenly mt-8">
            {socialLinks.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="transition-colors duration-300"
                style={{ color: TEXT_COLOR }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = HOVER_TEXT_COLOR)
                }
                onMouseLeave={(e) => (e.currentTarget.style.color = TEXT_COLOR)}
              >
                {item.icon}
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* Overlay for Mobile Menu */}
      {isOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black opacity-50 z-30"
          onClick={toggleMenu}
        ></div>
      )}
    </nav>
  );
};

export default Navbar;
