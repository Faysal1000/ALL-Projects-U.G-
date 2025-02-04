using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;
using TRPManagement.Annotation;
using TRPManagement.Entity_Framework;

namespace TRPManagement.DTOs
{
    public class ProgramDTO
    {
        public int ProgramId { get; set; }

        [Required]
        [MaxLength(150)]
        [UniqueProgramName("ChannelId")]
        public string ProgramName { get; set; }

        [Required]
        [Range(0.0,10.0)]
        public double TRPScore { get; set; }

        [Required]
        [ChannelShouldExist]
        public int ChannelId { get; set; }

        [Required]
        [TimeFormat("hh\\:mm\\:ss")]
        public System.TimeSpan AirTime { get; set; }
        public virtual Channel Channel { get; set; }
    }
}