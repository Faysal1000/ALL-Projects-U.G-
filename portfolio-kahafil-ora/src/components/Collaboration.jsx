const Collaboration = () => {
  return (
    <section className="bg-[#fff] flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-[5.2%] min-h-0">
      {/* Wrapper: fills section (flex-1), starts at top (justify-start), allows children to shrink (min-h-0) */}
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
        Creating Excellence Through Collaboration and Innovation
        <span className="text-[#9747FF]">.</span> {/* different color '.' */}
      </div>
    </section>
  );
};

export default Collaboration;
