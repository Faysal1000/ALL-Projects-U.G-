// src/config/brandLogos.js

import React from "react"; // needed because we include JSX nodes below
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

/*
  BRAND_LOGOS - central configuration for branded logos/icons used across the site.

  IMPORTANT (compatibility notes):
    - This file intentionally provides `icon` as a JSX node (e.g. <FaFacebook />)
      because your LogoItem component expects `item.icon` (and may call React.cloneElement).
    - If you later prefer to store component references (iconComponent: FaFacebook),
      update the renderer (LogoItem) to use React.createElement(item.iconComponent, {size}) instead.
    - Exports:
        - Named:  export const BRAND_LOGOS
        - Named:  export const logos (alias for older code)
        - Default: export default BRAND_LOGOS

  ITEM SHAPE (fields used by components):
    - name: string             // required - visible label & fallback alt/aria
    - icon: ReactNode          // preferred here (JSX node like <FaFacebook />)
    - image: string            // optional - URL or imported local asset (image takes priority over icon)
    - href: string             // optional - wrap item with anchor when rendering
    - size: number             // optional - preferred px size for the icon/image container
    - alt: string              // optional - alt text for images
    - ariaLabel: string        // optional - for anchors/buttons accessibility
    - priority: "high"|"normal"|"low" // optional - image loading preference

  IMAGE PATH GUIDELINES:
    - public/ folder: use absolute path like "/images/logo.png"
    - src/ imports: import at top and set image: importedVar (not used here, but supported)

  USAGE:
    - In your component file:
        import { BRAND_LOGOS } from "../config/brandLogos"; // adjust path as needed
      or
        import logos from "../config/brandLogos"; // default import (also works)

  MAINTENANCE:
    - Edit or reorder this file to update the marquee/partners list.
    - Keep items small (SVG/optimized PNG) to reduce layout shift.
*/

export const BRAND_LOGOS = [
  // Icons (JSX nodes). These are ready to be used by your LogoItem that expects item.icon.
  {
    name: "Gmail",
    icon: <SiGmail />,
    href: "mailto:hello@example.com",
    size: 40,
  },
  { name: "Facebook", icon: <FaFacebook />, size: 40 },
  { name: "Instagram", icon: <FaInstagram />, size: 40 },
  { name: "Twitter", icon: <FaTwitter />, size: 40 },
  { name: "LinkedIn", icon: <FaLinkedin />, size: 40 },
  { name: "YouTube", icon: <FaYoutube />, size: 40 },
  { name: "GitHub", icon: <FaGithub />, size: 40 },

  // Image example (public folder). Image takes precedence over icon in renderer.
  {
    name: "Custom1",
    image: "/images/custom-logo-1.png", // put file in public/images/custom-logo-1.png
    href: "https://example.com",
    size: 48,
    alt: "Custom Sponsor 1",
    priority: "normal",
  },

  // Continue icons
  { name: "WhatsApp", icon: <FaWhatsapp />, size: 40 },
  { name: "TikTok", icon: <SiTiktok />, size: 40 },
  { name: "Discord", icon: <SiDiscord />, size: 40 },
  { name: "Mailbox", icon: <GiMailbox />, size: 40 },
];

// alias for older code that may import `logos`
export const logos = BRAND_LOGOS;

// default export for convenience
export default BRAND_LOGOS;
