import Navbar from "./components/Navbar.jsx";
import HeroSection from "./components/HeroSection.jsx";
import AboutKahafilOra from "./components/AboutKahafilOra.jsx";
import Collaboration from "./components/Collaboration.jsx";
import LogoAnimation from "./components/LogoAnimation.jsx";
import LeadershipRoles from "./components/LeadershipRoles.jsx";
import BlogsAndArticles from "./components/BlogsAndArticles.jsx";
import Testimonial from "./components/Testimonial.jsx";
import Footer from "./components/Footer.jsx";

function Home() {
  return (
    <>
      <Navbar />
      <HeroSection />
      <AboutKahafilOra />
      <LogoAnimation />
      <Collaboration />
      <LeadershipRoles />
      <BlogsAndArticles />
      <Testimonial />
      <Footer />
    </>
  );
}

export default Home;
