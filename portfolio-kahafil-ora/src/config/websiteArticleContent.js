
//this controles main big news of the thoughts page
export const THOUGHT_SECTION_MAIN_NEWS = {
  title: "আইটি প্রফেশনাল মিটআপ",
  description:
    "দেশের আইটি খাতের ১৬০ জন আইটি প্রফেশনালকে বিশ্বমানের দক্ষতা উন্নয়ন প্রশিক্ষণ দেওয়া হয়েছে। সম্প্রতি প্রশিক্ষণ শেষে ঢাকা বিশ্ববিদ্যালয়ের আইবিএ বিভাগে ACMP 4.0 মিটআপ ২০২৪ অনুষ্ঠিত হয়।",
  date: "May 25, 2025",
  category: "Education, Social Impact",
  link: "https://www.facebook.com/events/123456789",
  image: "src/assets/thoughts_image.jpg",
};


/*
  Articles Object Structure:
  {
    id: Number,            // Unique identifier for each blog/article
    type: String,          // High-level category (e.g., "Cyber Security", "Cloud Migration")
    genre: String,         // Sub-category / section (e.g., "Blogs & Articles", "Public Speaking")
    date: String,          // Date of publication/event (format: "DD MMM YYYY")
    title: String,         // Headline or title of the blog/article
    img: String (URL),     // Image URL (can be external or local import)
    desc: String,          // Short description/summary
    link: String (URL),    // Link to full content (internal route or external URL)
  }
*/
/*
  IMPORTANT: Subject categories must always be chosen 
  from the following four options only:

  1. "Blogs & Articles"          → General blog posts & writeups
  2. "Public Speaking & Training"→ Talks, workshops, and training sessions
  3. "Recent Event Activities"   → Coverage of recent events & meetups
  4. "Social Activities"         → Social / community engagement activities
*/

/*
  IMPORTANT: If you want to add, remove, or update any subject category, 
  make the change ONLY in this array (ARTICLES_GENRES).  
  All content objects that use "genre" must reference one of these values 
  to keep the data consistent and prevent mismatches.
*/
export const ARTICLES_GENRES = [
  "Blogs & Articles",
  "Public Speaking & Training",
  "Recent Event Activities",
  "Social Activities",
];


// this controls all blogs and articles
export const BLOGS_AND_ARTICLES = [
  {
    id: 1,
    type: "Cyber Security",
    genre: "Blogs & Articles",
    date: "15 Aug 2025",
    title:
      "Faysal Ahmmed, A software engineer with excellent skills and academic scores",
    img: "https://placehold.co/200",
    desc: "Very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    type: "Cloud Migration",
    genre: "Public Speaking & Training",
    date: "15 Aug 2025",
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  },
  {
    id: 3,
    type: "Cybersecurity",
    genre: "Recent Event Activities",
    date: "15 Aug 2025",
    title: "Cybersecurity",
    img: "https://placehold.co/200",
    desc: "Protect data and systems with practical, audited security controls.",
    link: "#",
  },
  {
    id: 4,
    type: "DevOps & Automation",
    genre: "Social Activities",
    date: "15 Aug 2025",
    title: "DevOps & Automation",
    img: "https://placehold.co/200",
    desc: "Streamline delivery pipelines and reduce manual toil.",
    link: "/devops-automation",
  },
  {
    id: 5,
    type: "Digital Transformation",
    genre: "Social Activities",
    date: "15 Aug 2025",
    title: "Digital Transformation Strategy",
    img: "https://placehold.co/200",
    desc: "Helping businesses modernize operations through tailored digital adoption plans—enhancing efficiency, reducing costs, and boosting productivity.",
    link: "#",
  },
];

// Public Speaking & Training Articles
export const PUBLIC_SPEAKING_AND_TRAINING_ARTICLES =[
    {
    id: 1,
    type: "Cyber Security",
    genre: "Blogs & Articles",
    date: "15 Aug 2025",
    title:
      "Faysal Ahmmed, A software engineer with excellent skills and academic scores",
    img: "https://placehold.co/200",
    desc: "Very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    type: "Cloud Migration",
    genre: "Public Speaking & Training",
    date: "15 Aug 2025",
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  }
]

// recent event and activities
export const RECENT_EVENT_AND_ACTIVITIES =[
    {
    id: 1,
    type: "Cyber Security",
    genre: "Blogs & Articles",
    date: "15 Aug 2025",
    title:
      "Faysal Ahmmed, A software engineer with excellent skills and academic scores",
    img: "https://placehold.co/200",
    desc: "Very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    type: "Cloud Migration",
    genre: "Public Speaking & Training",
    date: "15 Aug 2025",
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  }
]


// Social activites objets
export const SOCIAL_ACTIVITIES =[
    {
    id: 1,
    type: "Cyber Security",
    genre: "Blogs & Articles",
    date: "15 Aug 2025",
    title:
      "Faysal Ahmmed, A software engineer with excellent skills and academic scores",
    img: "https://placehold.co/200",
    desc: "Very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    type: "Cloud Migration",
    genre: "Public Speaking & Training",
    date: "15 Aug 2025",
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  }
]


export const COLLABORATION_AND_INNOVATION_ARTICLES = [
  {
    id: 1,
    title: "Faysal Ahmmed",
    img: "https://placehold.co/200",
    desc: "very dedicated person in AI, deep learning along with research and Backend development.",
    link: "https://faysalahmmed-portfolio.vercel.app/",
  },
  {
    id: 2,
    title: "Cloud Migration",
    img: "https://placehold.co/200",
    desc: "Move legacy infrastructure to cloud for scalability and reliability.",
    link: "#",
  },
  {
    id: 3,
    title: "Cybersecurity",
    img: "https://placehold.co/200",
    desc: "Protect data and systems with practical, audited security controls.",
    link: "#",
  },
  {
    id: 4,
    title: "DevOps & Automation",
    img: "https://placehold.co/200",
    desc: "Streamline delivery pipelines and reduce manual toil.",
    link: "/devops-automation",
  },
  {
    id: 5,
    title: "Digital Transformation Strategy",
    img: "https://placehold.co/200",
    desc: "Helping businesses modernize operations through tailored digital adoption plans—enhancing efficiency, reducing costs, and boosting productivity.",
    link: "#",
  },
];




export default "Faysal Ahmmed wrote this codes @Email:faysalahmmed4200@gmail.com";