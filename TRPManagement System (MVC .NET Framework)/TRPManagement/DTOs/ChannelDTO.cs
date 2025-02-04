using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;
using TRPManagement.Annotation;
using TRPManagement.Entity_Framework;

namespace TRPManagement.DTOs
{
    public class ChannelDTO
    {
        public int ChannelId { get; set; }

        [Required]
        [MaxLength(100)]
        [ChannelShouldNotExist]
        public string ChannelName { get; set; }

        [Required]
        [DateLimitAndNoFutureDate]
        public int EstablishedYear { get; set; }

        [Required]
        [MaxLength(50)]
        public string Country { get; set; }
        public List<Program> Programs { get; set; }
    }
}