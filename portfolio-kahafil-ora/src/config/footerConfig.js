/* ------------------------------------------------------------------
   Footer Configuration - Single Source of Truth
   ------------------------------------------------------------------
   
   1. Social Links:
      - Defines the order and URLs of social media links that will render in the footer.
      - Each item must have:
        { 
          label: string,  // Name of the platform (e.g., "LinkedIn", "Instagram")
          href: string    // URL to the social profile
        }
      - The order in this array determines the order displayed in the UI.
  
   2. Footer Image Items:
      - Must contain exactly 6 objects.
      - Each item represents an image in the footer with an optional link.
      - Object structure:
        {
          image: string,                // Path to the image (can be local or URL)
          link: string,                 // Link when the image is clicked
          socialMediaType: string       // One of: "instagram" | "facebook" | "linkedin" | "website" | "none"
        }
      - Ensures consistent display of 6 images in the footer.

   NOTE:
      - If you need to update links or images, modify only this file.
      - Maintain exactly 6 items in `FOOTER_IMAGES_AND_LINKS` to avoid breaking the layout.
------------------------------------------------------------------ */

export const SOCIAL_LINKS = [
  { label: "X", href: "https://x.com" },
  { label: "LinkedIn", href: "https://linkedin.com/in/example" },
  { label: "YouTube", href: "https://example.com" },
  { label: "Facebook", href: "https://facebook.com/example" },
  { label: "Instagram", href: "https://instagram.com/example" },
];


export const FOOTER_IMAGES_AND_LINKS = [
  {
    image: "src/assets/Footer_Images/footer_image (1).png",
    link: "https://instagram.com/example1",
    socialMediaType: "instagram",
  },
  {
    image: "src/assets/Footer_Images/footer_image (2).png",
    link: "https://facebook.com/example2",
    socialMediaType: "facebook",
  },
  {
    image: "src/assets/Footer_Images/footer_image (3).png",
    link: "https://linkedin.com/in/example3",
    socialMediaType: "linkedin",
  },
  {
    image: "src/assets/Footer_Images/footer_image (4).png",
    link: "https://example.com/4",
    socialMediaType: "website",
  },
  {
    image: "src/assets/Footer_Images/footer_image (5).png",
    link: "https://instagram.com/example5",
    socialMediaType: "instagram",
  },
  {
    image: "src/assets/Footer_Images/footer_image (6).png",
    link: "https://example.com/6",
    socialMediaType: "website",
  },
];


export default "Faysal Ahmmed wrote this codes @Email:faysalahmmed4200@gmail.com";