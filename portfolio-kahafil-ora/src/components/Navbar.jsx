import { useState } from "react";
import { useEffect } from "react";
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
  const [isScrolled, setIsScrolled] = useState(false);

  // Check if the user has scrolled down the page
  // to change the navbar style
  // This effect runs once on mount and sets up an event listener
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10); 
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);


  // Lock scroll when mobile menu is open
  // This effect runs whenever isOpen changes.
  // It saves the current scroll position when the menu opens,
  // and restores it when the menu closes.
  // It also cleans up the styles when the component unmounts.
  // This prevents the background from scrolling when the menu is open.
  useEffect(() => {
    let scrollY = 0;

    if (isOpen) {
      // Save current scroll position
      scrollY = window.scrollY;
      
      // Lock scroll and freeze at top
      document.body.style.position = "fixed";
      document.body.style.top = `-${scrollY}px`;
      document.body.style.overflow = "hidden";
      document.body.style.width = "100%";
    } else {
      // Restore scroll position
      const y = document.body.style.top;
      document.body.style.position = "";
      document.body.style.top = "";
      document.body.style.overflow = "";
      document.body.style.width = "";
      window.scrollTo(0, parseInt(y || "0") * -1);
    }

    return () => {
      // In case component unmounts while locked
      document.body.style.position = "";
      document.body.style.top = "";
      document.body.style.overflow = "";
      document.body.style.width = "";
    };
  }, [isOpen]);

  // Handle window resize to close mobile menu
  // to close the mobile menu if the window is resized to a larger width.
  // It cleans up the event listener on unmount.
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setIsOpen(false); // Close the mobile menu if window is larger than md breakpoint
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (

      <nav
        className={`bg-cover bg-no-repeat sticky top-0 z-50 transition-all duration-300 ${isScrolled ? "backdrop-blur-md shadow-xs py-0" : "py-2"}`}
        style={{
          backgroundImage: `url(${TEXTURE})`
        }}
      >
      <div className="container mx-auto flex items-center">
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
            className="p-3 focus:outline-none"
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
