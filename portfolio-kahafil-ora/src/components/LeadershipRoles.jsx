const LeadershipRoles = () => {
  return (
    <section className="bg-[#fff] pt-15 md:pt-0 flex-1 flex py-[5.2%] px-[12.5%] flex-col justify-start items-start gap-[5.2%] min-h-0">
      {/* Wrapper: fills section (flex-1), starts at top (justify-start), allows children to shrink (min-h-0) */}
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-[700] lowercase leading-normal">
        leadership roles
        <span className="text-[#9747FF]">.</span> {/* different color '.' */}
      </div>

      {/* Divider */}
      <div className="self-stretch py-[4.2%]">
        <div className="border-t border-[#444]"></div>
      </div>

      <div className="flex flex-col lg:flex-row justify-between items-start self-stretch gap-6">
        {/* Left text */}
        <div className="w-full lg:w-[38.49%] text-[#444] font-['Poppins'] text-xl lg:text-2xl font-light leading-normal">
          I've worked with companies and clients, both in agency settings. I
          enjoy collaborating with clients who appreciate the importance of good
          design.
        </div>

        {/* Scrollable table */}
        <div className="w-full lg:w-[59.51%] pb-[3%] overflow-x-auto">
          <div className="min-w-[600px] flex flex-col items-start">
            {/* Row 1 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                Goinnovior Limited
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2016-Present
              </div>
            </div>

            {/* Row 2 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                360D Soul Limited
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2023-Present
              </div>
            </div>

            {/* Row 3 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                CodeInnovior
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2020-Present
              </div>
            </div>

            {/* Row 4 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                Skylark Soft Limited
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Head of Businesses
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2022-2024
              </div>
            </div>

            {/* Row 5 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                Impress Group
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Head of Information Technology Operations.
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2018-2022
              </div>
            </div>

            {/* Row 6 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                Next IT Ltd.
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Founder & Managing Director
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2016-2018
              </div>
            </div>

            {/* Row 7 */}
            <div className="grid grid-cols-3 py-[4.14%] items-center border-b border-[#444] w-full">
              <div className="text-[#444] font-[Amiri] text-xl font-normal leading-normal">
                MASCO Group
              </div>
              <div className="text-[#444] text-center font-[Amiri] text-sm font-normal leading-normal">
                Head of Information Technology Department.
              </div>
              <div className="text-[#444] text-right font-[Fragment_Mono] text-sm font-normal leading-normal">
                2010-2016
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LeadershipRoles;
