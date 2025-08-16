import Navbar from "./components/Navbar.jsx";
import HeroSection from "./components/HeroSection.jsx";
import AboutSection from "./components/AboutSection.jsx";
import Collaboration from "./components/Collaboration.jsx";
import LogoAnimation from "./components/LogoAnimation.jsx";
import LeadershipRoles from "./components/LeadershipRoles.jsx";
import BlogsAndArticles from "./components/BlogsAndArticles.jsx";
import Testimonial from "./components/Testimonial.jsx";
function Home() {
  return (
    <>
      <Navbar />
      <HeroSection />
      <AboutSection />
      <LogoAnimation />
      <Collaboration />
      <LeadershipRoles />
      <BlogsAndArticles />
      <Testimonial />
    </>
  );
}

export default Home;
